"""
Stationary hazard collision (Town04).

A stationary vehicle sits as a hazard in the road; a firetruck drives into
it at a constant throttle while ~28 background traffic vehicles drive
normally nearby. An ego vehicle drives through and observes. Sensor data for
one chosen vehicle (ego, a colliding vehicle, or any background vehicle) is
recorded via sensor_recorder.SensorRecorder -- see the repo README for how
to pick which vehicle/sensor type to record.
"""

import os
import threading
import time
import traceback

import py_trees
import carla

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import KeepVelocity
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from sensor_recorder import SensorRecorder, list_vehicles


def EndAfterCollision(collided_ids, delay=0.5, name="EndAfterCollision"):
    """
    py_trees behavior: RUNNING until `collided_ids` (a shared, mutable set
    populated by the firetruck's collision handler) becomes non-empty, then
    RUNNING for `delay` more simulated seconds, then SUCCESS. Pair with a
    root Parallel using ParallelPolicy.SUCCESS_ON_ONE so the scenario ends
    as soon as this succeeds -- without needing every other perpetually-
    RUNNING behavior (KeepVelocity, RecordSensorsTick, ...) to also succeed.
    """
    class _EndAfterCollision(py_trees.behaviour.Behaviour):
        def __init__(self):
            super(_EndAfterCollision, self).__init__(name)
            self._collided_ids = collided_ids
            self._delay = delay
            self._collision_time = None

        def update(self):
            if self._collision_time is None:
                if not self._collided_ids:
                    return py_trees.common.Status.RUNNING
                self._collision_time = GameTime.get_time()
            if GameTime.get_time() - self._collision_time >= self._delay:
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING

    return _EndAfterCollision()


def RecordSensorsTick(scenario, name="RecordSensorsTick"):
    """
    py_trees behavior: calls scenario.recorder.on_tick() once per scenario
    tick. Reads scenario.recorder fresh every tick (instead of capturing it
    once at construction time) because the colliding vehicles in this
    scenario are only activated a few seconds in -- scenario._try_setup_recorder()
    keeps attempting setup here every tick until its target vehicle exists.
    """
    class _Tick(py_trees.behaviour.Behaviour):
        def __init__(self):
            super(_Tick, self).__init__(name)
            self._scenario = scenario

        def update(self):
            # Broad try/except: an exception raised from inside a behavior's
            # update() can otherwise be silently swallowed by py_trees /
            # scenario_runner (no traceback, the scenario just quietly never
            # progresses this behavior again), hiding the real cause of a
            # "recording never started" failure.
            try:
                if self._scenario.recorder is None and not self._scenario.recorder_setup_failed:
                    self._scenario._try_setup_recorder()
                if self._scenario.recorder is not None:
                    self._scenario.recorder.on_tick()
            except Exception as e:
                if not self._scenario.recorder_setup_failed:
                    self._scenario.recorder_setup_failed = True
                    print("[SensorRecorder] ALERT - RecordSensorsTick.update() crashed: {}: {}".format(
                        type(e).__name__, e))
                    traceback.print_exc()
            return py_trees.common.Status.RUNNING

    return _Tick()


class StationaryHazardCollision(BasicScenario):
    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True):
        self.timeout = 120
        self.surrounding_vehicles = []
        self.colliding_vehicles = []
        self.collision_sensors = []
        self.colliding_points_config = []
        self._throttle_threads = []
        self._stop_threads_event = threading.Event()

        # Populated by _stop_vehicle_on_collision() -- read by EndAfterCollision
        # to end the scenario a fixed delay after the firetruck hits something.
        self.collided_vehicle_ids = set()

        # Set True by _activate_firetruck() once its delay fires.
        self.firetruck_activated = False

        # ---------- SENSOR RECORDING state ----------
        # Setup is deferred (see _try_setup_recorder) because the colliding
        # vehicles are only activated a few seconds into the run, so they
        # don't exist yet when _initialize_actors() runs.
        self.recorder = None
        self.recorder_setup_failed = False
        self._recorder_setup_attempts = 0
        self._record_vehicle_index = None
        self._record_role_name = None
        self._record_sensor_type = "cameras"
        self._record_output_dir = "sensor_output"
        self._record_fps = 20.0

        super(StationaryHazardCollision, self).__init__(
            name="StationaryHazardCollision",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            criteria_enable=criteria_enable
        )

    def _setup_scenario_trigger(self, config):
        # scenario_parser.py auto-populates config.trigger_points with the
        # ego's OWN spawn transform for every plain <scenario> XML (this one
        # included). BasicScenario's default _setup_scenario_trigger() would
        # then return InTimeToArrivalToLocation(ego, 2.0, ego's own spawn
        # point) as the first child of the outer behavior Sequence -- a
        # condition that needs the ego to already be moving toward a point
        # it's currently sitting motionless on, so it never succeeds and the
        # Sequence never advances to _create_behavior()'s tree at all.
        return None

    def _initialize_actors(self, config):
        world = CarlaDataProvider.get_world()
        blueprint_library = world.get_blueprint_library()
        tm = CarlaDataProvider.get_client().get_trafficmanager()
        tm_port = tm.get_port()

        self.colliding_points_config = [
            {"x": 149.74, "y": -173.30, "z": 0.60, "yaw": -179.67, "model": "vehicle.nissan.micra", "color": "0,0,0"},
            {"x": 227.07, "y": -172.86, "z": 0.56, "yaw": -179.67, "model": "vehicle.carlamotors.firetruck", "color": "0,0,255"},
        ]

        surround_points = [
            {"x": 104.94, "y": -173.56, "z": 0.60, "yaw": -179.67, "model": "vehicle.toyota.prius", "color": "234,0,0"},
            {"x": 201.20, "y": -283.47, "z": 0.48, "yaw": 91.31, "model": "vehicle.audi.tt", "color": "16,44,21"},
            {"x": 284.55, "y": -172.54, "z": 0.54, "yaw": -179.67, "model": "vehicle.lincoln.mkz_2017", "color": "16,16,16"},
            {"x": 284.78, "y": -169.04, "z": 0.54, "yaw": 0.33, "model": "vehicle.mercedes.coupe_2020", "color": "73,0,0"},
            {"x": 285.91, "y": -121.88, "z": 0.58, "yaw": -179.08, "model": "vehicle.mini.cooper_s_2021", "color": "0,0,0"},
            {"x": 314.29, "y": -146.64, "z": 0.28, "yaw": -89.49, "model": "vehicle.carlamotors.firetruck", "color": "234,0,0"},
            {"x": 233.21, "y": -307.65, "z": 0.38, "yaw": 0.59, "model": "vehicle.nissan.micra", "color": "28,0,46"},
            {"x": 226.96, "y": -311.22, "z": 0.38, "yaw": -179.41, "model": "vehicle.citroen.c3", "color": "217,217,217"},
            {"x": 255.27, "y": -280.21, "z": 0.58, "yaw": 90.18, "model": "vehicle.dodge.charger_2020", "color": "211,142,0"},
            {"x": 229.87, "y": -246.03, "z": 0.28, "yaw": -0.39, "model": "vehicle.audi.tt", "color": "168,0,27"},
            {"x": -515.25, "y": 240.96, "z": 0.28, "yaw": 89.87, "model": "vehicle.nissan.patrol", "color": "183,187,162"},
            {"x": -511.76, "y": 242.55, "z": 0.28, "yaw": 89.87, "model": "vehicle.volkswagen.t2", "color": "105,103,0"},
            {"x": -508.25, "y": 240.94, "z": 0.28, "yaw": 89.87, "model": "vehicle.toyota.prius", "color": "255,0,0"},
            {"x": -504.76, "y": 242.53, "z": 0.28, "yaw": 89.87, "model": "vehicle.toyota.prius", "color": "234,0,0"},
            {"x": -493.23, "y": 177.65, "z": 0.28, "yaw": -89.64, "model": "vehicle.lincoln.mkz_2017", "color": "16,16,16"},
            {"x": -508.35, "y": 198.44, "z": 0.28, "yaw": 89.87, "model": "vehicle.bmw.grandtourer", "color": "0,0,0"},
            {"x": -504.86, "y": 200.03, "z": 0.28, "yaw": 89.87, "model": "vehicle.volkswagen.t2", "color": "73,12,12"},
            {"x": -514.11, "y": 158.82, "z": 0.28, "yaw": 90.36, "model": "vehicle.audi.tt", "color": "16,44,21"},
            {"x": -510.62, "y": 159.84, "z": 0.28, "yaw": 90.36, "model": "vehicle.lincoln.mkz_2017", "color": "16,16,16"},
            {"x": -507.11, "y": 158.86, "z": 0.28, "yaw": 90.36, "model": "vehicle.mercedes.coupe_2020", "color": "73,0,0"},
            {"x": -503.62, "y": 159.89, "z": 0.28, "yaw": 90.36, "model": "vehicle.mini.cooper_s_2021", "color": "215,88,0"},
            {"x": 37.25, "y": -170.44, "z": 0.60, "yaw": 0.33, "model": "vehicle.ford.crown", "color": "255,185,0"},
            {"x": 83.19, "y": -170.19, "z": 0.60, "yaw": 0.33, "model": "vehicle.ford.ambulance", "color": "231,231,231"},
            {"x": 202.76, "y": -198.99, "z": 0.78, "yaw": -88.69, "model": "vehicle.bmw.grandtourer", "color": "0,0,0"},
            {"x": 203.30, "y": -222.19, "z": 0.78, "yaw": -88.69, "model": "vehicle.volkswagen.t2", "color": "73,12,12"},
            {"x": 199.53, "y": -210.40, "z": 0.78, "yaw": 91.31, "model": "vehicle.ford.crown", "color": "255,185,0"},
            {"x": 199.04, "y": -189.10, "z": 0.78, "yaw": 91.31, "model": "vehicle.tesla.model3", "color": "255,0,0"},
            {"x": 105.29, "y": -170.06, "z": 0.60, "yaw": 0.33, "model": "vehicle.mini.cooper_s", "color": "0,255,0"},
        ]

        # Colliding vehicles spawn first (right after the ego) so they land
        # at low, stable indices N=1,2 -- matching the other recorded
        # scenarios. spawn_colliding_vehicles() only spawns them stationary
        # and wires up the collision sensor; the firetruck doesn't actually
        # start moving until _activate_firetruck() fires after a short
        # delay, so the hazard interaction still begins a few seconds in.
        self.spawn_colliding_vehicles()

        print("Spawning surrounding vehicles...")
        for sp in surround_points:
            bp = blueprint_library.find(sp['model'])
            if bp.has_attribute('color'):
                bp.set_attribute('color', sp['color'])

            transform = carla.Transform(carla.Location(x=sp['x'], y=sp['y'], z=sp['z']),
                                         carla.Rotation(yaw=sp['yaw']))
            veh = world.try_spawn_actor(bp, transform)
            if veh:
                veh.set_autopilot(True, tm_port)
                tm.vehicle_percentage_speed_difference(veh, 80.0)
                self.surrounding_vehicles.append(veh)

                lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam | carla.VehicleLightState.Fog
                veh.set_light_state(carla.VehicleLightState(lights))
            else:
                print("[SpawnFail] surrounding vehicle '{}' at ({}, {}, {}) yaw={} -- spot blocked/invalid".format(
                    sp['model'], sp['x'], sp['y'], sp['z'], sp['yaw']))

        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            carla.Location(x=111.64, y=-175.61, z=40.23),
            carla.Rotation(pitch=-38.39, yaw=-2.02, roll=0.00)
        ))

        # ---------- SENSOR RECORDING: pick ONE vehicle + ONE sensor type ----------
        # Selectable at launch time via environment variables, no code edits needed:
        #   RECORD_VEHICLE_INDEX -> 0-based ordinal into the vehicle list printed
        #                           at scenario start (spawn order: ego, then
        #                           N=1 the stationary nissan micra hazard,
        #                           N=2 the firetruck that drives into it,
        #                           then N=3.. the surrounding vehicles).
        #   RECORD_VEHICLE_ROLE_NAME -> select by role name instead of index
        #   RECORD_SENSOR_TYPE   -> "cameras", "lidar", or "both" (default "cameras")
        #   RECORD_OUTPUT_DIR    -> output root (default "sensor_output")
        #   RECORD_FPS           -> must match world fixed_delta_seconds (default 20.0)
        # Actual setup happens lazily in _try_setup_recorder(), called every
        # tick from RecordSensorsTick, as a safety net -- both colliding
        # vehicles and all surrounding vehicles already exist by the time
        # this method returns, so in practice it succeeds on the first tick.
        record_vehicle_index_env = os.environ.get("RECORD_VEHICLE_INDEX")
        self._record_vehicle_index = int(record_vehicle_index_env) if record_vehicle_index_env is not None else None
        self._record_role_name = os.environ.get("RECORD_VEHICLE_ROLE_NAME")
        if self._record_vehicle_index is None and self._record_role_name is None:
            self._record_vehicle_index = 0  # default: record the ego vehicle
        self._record_sensor_type = os.environ.get("RECORD_SENSOR_TYPE", "cameras")
        self._record_output_dir = os.environ.get("RECORD_OUTPUT_DIR", "sensor_output")
        self._record_fps = float(os.environ.get("RECORD_FPS", "20.0"))

    def _try_setup_recorder(self):
        """
        Attempt to create+attach the SensorRecorder. Called every tick until
        it succeeds -- needed because a vehicle selected by index/role_name
        may not exist yet.
        """
        self._recorder_setup_attempts += 1
        world = CarlaDataProvider.get_world()
        vehicles = list_vehicles(world)

        if self._record_role_name is not None:
            target_ready = any(v["role_name"] == self._record_role_name for v in vehicles)
        else:
            target_ready = len(vehicles) > self._record_vehicle_index
        if not target_ready:
            return  # keep waiting for the target vehicle to spawn

        print("[SensorRecorder] Vehicles available for recording (0..{}):".format(len(vehicles) - 1))
        for v in vehicles:
            print("    index={:<3d} id={:<4d} role_name='{}' type={}".format(
                v["index"], v["id"], v["role_name"], v["type_id"]))

        try:
            self.recorder = SensorRecorder(
                world=world,
                output_dir=self._record_output_dir,
                scenario_name="StationaryHazardCollision",
                fps=self._record_fps,
                sensor_type=self._record_sensor_type,
                vehicle_index=self._record_vehicle_index,
                vehicle_role_name=self._record_role_name,
            )
            self.recorder.setup_sensors()
            print("[SensorRecorder] Recording vehicle_label='{}' sensor_type='{}' -> {}".format(
                self.recorder.vehicle_label, self._record_sensor_type, self.recorder.output_dir))
        except Exception as e:
            self.recorder = None
            self.recorder_setup_failed = True
            print("[SensorRecorder] ALERT - recording disabled: {}: {}".format(type(e).__name__, e))
            traceback.print_exc()

    def spawn_colliding_vehicles(self):
        """
        Spawns both colliding vehicles immediately (stationary) so they get
        low, stable indices (N=1,2) right after the ego. The firetruck stays
        parked -- autopilot off, no throttle applied -- until
        _activate_firetruck() fires a few seconds later.
        """
        print("Spawning colliding vehicles (firetruck activates after a short delay)...")
        world = CarlaDataProvider.get_world()
        blueprint_library = world.get_blueprint_library()
        tm = CarlaDataProvider.get_client().get_trafficmanager()
        tm_port = tm.get_port()

        for cp in self.colliding_points_config:
            bp = blueprint_library.find(cp['model'])
            if bp.has_attribute('color'):
                bp.set_attribute('color', cp['color'])

            transform = carla.Transform(
                carla.Location(x=cp['x'], y=cp['y'], z=cp['z']),
                carla.Rotation(yaw=cp['yaw'])
            )
            veh = world.try_spawn_actor(bp, transform)
            if not veh:
                print("[SpawnFail] colliding vehicle '{}' at ({}, {}, {}) yaw={} -- spot blocked/invalid".format(
                    cp['model'], cp['x'], cp['y'], cp['z'], cp['yaw']))
                continue

            self.colliding_vehicles.append(veh)
            veh.set_autopilot(False, tm_port)
            tm.ignore_lights_percentage(veh, 100)
            tm.auto_lane_change(veh, False)

            if cp['model'] == "vehicle.carlamotors.firetruck":
                collision_bp = blueprint_library.find('sensor.other.collision')
                collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=veh)
                collision_sensor.listen(lambda event: self._stop_vehicle_on_collision(veh))
                self.collision_sensors.append(collision_sensor)

                print("Spawned stationary firetruck (activates to throttle {} after delay)".format(
                    self.FIRETRUCK_BASE_THROTTLE))
                threading.Timer(6.0, self._activate_firetruck, args=(veh,)).start()
            else:
                # Stationary hazard vehicle. Disabling physics makes it a
                # fixed, immovable hazard that won't get shoved out of place
                # by the firetruck's impact -- but disabling it IMMEDIATELY
                # on spawn would freeze it at its exact spawn Z, before
                # gravity has a chance to settle it onto the actual road
                # surface (making it appear to float). Give it a brief
                # moment to settle first.
                threading.Timer(1.5, veh.set_simulate_physics, args=(False,)).start()
                print("Spawned stationary hazard vehicle: {}".format(cp['model']))

    FIRETRUCK_BASE_THROTTLE = 0.2

    def _activate_firetruck(self, vehicle):
        """
        Starts the firetruck moving at a plain constant throttle. Called
        after a delay (see spawn_colliding_vehicles()) so the vehicle itself
        can still spawn immediately for a stable low index.
        """
        if not vehicle or not vehicle.is_alive:
            return
        print("Activating firetruck: applying throttle {}.".format(self.FIRETRUCK_BASE_THROTTLE))
        self.firetruck_activated = True

        control = carla.VehicleControl()
        control.hand_brake = False
        control.throttle = self.FIRETRUCK_BASE_THROTTLE
        vehicle.apply_control(control)

        control_thread = threading.Thread(target=self._apply_throttle_loop, args=(vehicle,))
        control_thread.start()
        self._throttle_threads.append(control_thread)

    def _apply_throttle_loop(self, vehicle):
        """Continuously applies throttle to the vehicle until stopped or scenario ends."""
        while not self._stop_threads_event.is_set():
            try:
                if not vehicle or not vehicle.is_alive:
                    break
                if hasattr(vehicle, 'stop_flag') and vehicle.stop_flag:
                    break

                control = carla.VehicleControl()
                control.throttle = self.FIRETRUCK_BASE_THROTTLE
                control.hand_brake = False
                vehicle.apply_control(control)
                time.sleep(0.05)
            except RuntimeError:
                break

    def _stop_vehicle_on_collision(self, vehicle):
        if vehicle.id in self.collided_vehicle_ids:
            return  # already handled -- avoid re-triggering every tick while stuck in contact
        self.collided_vehicle_ids.add(vehicle.id)
        print("Firetruck collision detected, stopping vehicle.")
        vehicle.stop_flag = True  # custom flag read by _apply_throttle_loop to exit
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

    def _create_behavior(self):
        # SUCCESS_ON_ONE (not SUCCESS_ON_ALL): EgoKeepVelocity and
        # RecordSensorsTick run forever and never succeed on their own, so
        # the scenario has to end as soon as ANY one child succeeds --
        # either EndAfterCollision (0.5s after the firetruck hits something)
        # or the ego completing its drive distance as a fallback.
        root = py_trees.composites.Parallel("Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(KeepVelocity(self.ego_vehicles[0], target_velocity=10.0, name="EgoKeepVelocity"))
        root.add_child(RecordSensorsTick(self))
        root.add_child(EndAfterCollision(self.collided_vehicle_ids, delay=0.5))
        root.add_child(DriveDistance(self.ego_vehicles[0], distance=500.0, name="ScenarioCompletion"))
        return root

    def _create_test_criteria(self):
        return [CollisionTest(self.ego_vehicles[0])]

    def remove_all_actors(self):
        # basic_scenario.py never calls a method named "end_scenario" or
        # relies on __del__ timing -- remove_all_actors() is the actual hook
        # scenario_runner.py invokes once the run finishes.
        self._stop_threads_event.set()
        for thread in self._throttle_threads:
            thread.join()

        if self.recorder is not None:
            try:
                self.recorder.on_tick()  # flush the last buffered frame
            except Exception:
                pass
            self.recorder.destroy()
            self.recorder = None
        elif not self.recorder_setup_failed:
            # _try_setup_recorder() never saw its target vehicle exist for
            # this entire run -- e.g. RECORD_VEHICLE_INDEX was higher than
            # the number of vehicles that actually ended up spawned, or a
            # RECORD_VEHICLE_ROLE_NAME was never assigned to anything.
            print("[SensorRecorder] ALERT - recording never started: the requested "
                  "vehicle_index='{}' / role_name='{}' never existed during this run. "
                  "Check the printed vehicle count against your RECORD_VEHICLE_INDEX.".format(
                      self._record_vehicle_index, self._record_role_name))

        super(StationaryHazardCollision, self).remove_all_actors()

    def __del__(self):
        """
        Cleanup safety net -- remove_all_actors() above is the real hook the
        framework calls; this just covers interpreter-exit edge cases.
        """
        try:
            self.remove_all_actors()
        except Exception:
            pass
