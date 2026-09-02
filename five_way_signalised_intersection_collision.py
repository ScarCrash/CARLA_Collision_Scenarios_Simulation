"""
5-way signalised intersection collision (Town03).

Two vehicles collide head-on/broadside at a 5-way signalised intersection,
surrounded by ~30 background traffic vehicles. An ego vehicle drives through
and observes. Sensor data for one chosen vehicle (ego, a colliding vehicle,
or any background vehicle) is recorded via sensor_recorder.SensorRecorder --
see the repo README for how to pick which vehicle/sensor type to record.
"""

import os
import random

import py_trees
import carla
from carla import VehicleControl

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import KeepVelocity
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from sensor_recorder import SensorRecorder, list_vehicles


def _spawn_with_retry(world, bp, transform, retries=1, jitter_attempts=6, jitter_radius=3.0,
                       fallback_points=None, used_fallback_indices=None):
    """
    try_spawn_actor() can spuriously report a spot as blocked right after a
    world reload, while streamed map colliders are still settling -- a tick
    + retry clears that. For a spot that's simply not on drivable road at
    all, prefer falling back to one of the map's own guaranteed-valid
    recommended spawn points (world.get_map().get_spawn_points()) near the
    scene -- those carry a position AND heading that are correct together.
    Raw position jitter (kept as a last resort) reuses the ORIGINAL
    rotation at a nudged position, which can end up facing the wrong way if
    the nudge lands on a different lane/road than the original point was on.
    """
    veh = world.try_spawn_actor(bp, transform)
    if veh:
        return veh
    for _ in range(retries):
        if CarlaDataProvider.is_sync_mode():
            world.tick()
        else:
            world.wait_for_tick()
        veh = world.try_spawn_actor(bp, transform)
        if veh:
            return veh

    base_loc = transform.location
    if fallback_points:
        candidates = sorted(
            range(len(fallback_points)),
            key=lambda i: base_loc.distance(fallback_points[i].location),
        )
        for i in candidates:
            if used_fallback_indices is not None and i in used_fallback_indices:
                continue
            veh = world.try_spawn_actor(bp, fallback_points[i])
            if veh:
                if used_fallback_indices is not None:
                    used_fallback_indices.add(i)
                return veh

    for _ in range(jitter_attempts):
        dx = random.uniform(-jitter_radius, jitter_radius)
        dy = random.uniform(-jitter_radius, jitter_radius)
        jittered = carla.Transform(
            carla.Location(x=base_loc.x + dx, y=base_loc.y + dy, z=base_loc.z + 0.5),
            transform.rotation,
        )
        veh = world.try_spawn_actor(bp, jittered)
        if veh:
            return veh
    return None


def EndAfterCollision(collided_ids, delay=2.0, name="EndAfterCollision"):
    """
    py_trees behavior: RUNNING until `collided_ids` (a shared, mutable set
    populated by a collision handler) becomes non-empty, then RUNNING for
    `delay` more simulated seconds, then SUCCESS. Pair with a root Parallel
    using ParallelPolicy.SUCCESS_ON_ONE so the scenario ends as soon as this
    succeeds -- without needing every other perpetually-RUNNING behavior
    (KeepVelocity, ManualThrottle, ...) to also reach SUCCESS.
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


def RecordSensorsTick(recorder, name="RecordSensorsTick"):
    """py_trees behavior: calls recorder.on_tick() once per scenario tick."""
    class _Tick(py_trees.behaviour.Behaviour):
        def __init__(self):
            super(_Tick, self).__init__(name)
            self._recorder = recorder

        def update(self):
            self._recorder.on_tick()
            return py_trees.common.Status.RUNNING

    return _Tick()


def ManualThrottle(actor, name="ManualThrottle"):
    """py_trees behavior: applies a neutral control every tick to a colliding vehicle."""
    class Throttle(py_trees.behaviour.Behaviour):
        def __init__(self):
            super(Throttle, self).__init__(name)
            self._actor = actor

        def update(self):
            control = VehicleControl(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, gear=1)
            self._actor.apply_control(control)
            return py_trees.common.Status.RUNNING

    return Throttle()


class FiveWaySignalisedIntersectionCollision(BasicScenario):
    """
    Vehicle order (spawn order == vehicle index):
      N=0            ego
      N=1, N=2       the 2 colliding vehicles
      N=3 .. N=30    ~28 curated background vehicles + top-up to 30 total
    """

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True):
        self.timeout = 60
        self.surrounding_vehicles = []
        self.colliding_vehicles = []
        self.collision_sensors = []
        self.recorder = None
        self._collided_vehicle_ids = set()
        super(FiveWaySignalisedIntersectionCollision, self).__init__(
            name="FiveWaySignalisedIntersectionCollision",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            criteria_enable=criteria_enable)

    def _setup_scenario_trigger(self, config):
        # BasicScenario's default trigger is InTimeToArrivalToLocation(ego, 2.0,
        # ego's own spawn point). Its update() falls back to time_to_arrival=inf
        # whenever the ego's velocity is ~0 -- and the ego starts at velocity 0
        # with autopilot off, only gaining speed from KeepVelocity *inside* the
        # very behavior tree this trigger gates. That's a deadlock: it can never
        # succeed, so _create_behavior()'s whole tree (including the sensor
        # recorder) never ticks. This scenario doesn't need staged/route-based
        # triggering, so skip it and start the real behavior immediately.
        return None

    def _initialize_actors(self, config):
        if hasattr(config, "weather") and config.weather:
            CarlaDataProvider.get_world().set_weather(config.weather)

        world = CarlaDataProvider.get_world()
        blueprint_library = world.get_blueprint_library()
        tm = CarlaDataProvider.get_client().get_trafficmanager()
        tm_port = tm.get_port()

        # ---------- CLEANUP: self-heal from a previous interrupted/crashed run ----------
        # If a prior run didn't reach remove_all_actors() (Ctrl+C, crash,
        # --reloadWorld not passed), its vehicles are still sitting at these
        # same hardcoded spawn points and block try_spawn_actor() below.
        # Destroy anything that isn't this run's ego before spawning, so
        # every run starts from a clean, deterministic scene.
        ego_ids = {v.id for v in self.ego_vehicles}
        leftover = [v for v in world.get_actors().filter('vehicle.*') if v.id not in ego_ids]
        if leftover:
            print("[Cleanup] Destroying {} leftover vehicle(s) from a previous run...".format(len(leftover)))
            for v in leftover:
                try:
                    v.destroy()
                except RuntimeError:
                    pass

        # ---------- SETTLE: let streamed map geometry/colliders catch up ----------
        # Right after --reloadWorld, Town03's static colliders can still be
        # settling; spawning immediately can spuriously report spawn points
        # as blocked.
        for _ in range(10):
            if CarlaDataProvider.is_sync_mode():
                world.tick()
            else:
                world.wait_for_tick()

        # ---------- COLLIDING VEHICLES ----------
        colliding_points = [
            {"x": -54.75, "y": -2.90, "z": 0.28, "yaw": -179.71, "speed": 4.0,
             "model": "vehicle.tesla.model3", "color": "255,0,0"},
            {"x": -84.96, "y": -23.28, "z": 0.28, "yaw": 89.84, "speed": 3.0,
             "model": "vehicle.tesla.model3", "color": "0,0,0"},
        ]

        for cp in colliding_points:
            bp = blueprint_library.find(cp['model'])
            if bp.has_attribute('color'):
                bp.set_attribute('color', cp['color'])
            transform = carla.Transform(
                carla.Location(x=cp['x'], y=cp['y'], z=cp['z']),
                carla.Rotation(yaw=cp['yaw']))
            veh = _spawn_with_retry(world, bp, transform)
            if not veh:
                print("[SpawnFail] colliding vehicle '{}' at ({}, {}, {}) yaw={} -- spot blocked/invalid".format(
                    cp['model'], cp['x'], cp['y'], cp['z'], cp['yaw']))
                continue
            veh.set_simulate_physics(True)
            veh.set_autopilot(False, tm_port)
            tm.ignore_vehicles_percentage(veh, 100.0)
            tm.ignore_lights_percentage(veh, 100.0)
            forward = transform.rotation.get_forward_vector()
            speed = cp.get('speed', 3.0)
            vel_vec = carla.Vector3D(forward.x * speed, forward.y * speed, forward.z * speed)
            veh.set_target_velocity(vel_vec)
            veh.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            col_bp = blueprint_library.find('sensor.other.collision')
            col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=veh)
            col_sensor.listen(lambda event, v=veh: self._on_collision(v))
            self.collision_sensors.append(col_sensor)
            self.colliding_vehicles.append(veh)

        # ---------- EGO ----------
        ego = self.ego_vehicles[0]
        ego.set_autopilot(False, tm_port)
        ego.set_simulate_physics(True)

        # ---------- SURROUNDING VEHICLES ----------
        # Two are given distinctive role_names for convenient selection:
        #   "lidar_target"  -- vehicle.ford.crown
        #   "camera_target" -- vehicle.ford.ambulance
        surround_points = [
            {"x": -78.12, "y": -81.24, "z": 0.28, "yaw": -90.16, "model": "vehicle.toyota.prius", "color": "234,0,0"},
            {"x": -74.62, "y": -95.05, "z": 0.28, "yaw": -90.16, "model": "vehicle.lincoln.mkz_2017", "color": "16,16,16"},
            {"x": -77.99, "y": -30.24, "z": 0.28, "yaw": -90.16, "model": "vehicle.bmw.grandtourer", "color": "0,0,0"},
            {"x": -74.49, "y": -44.05, "z": 0.28, "yaw": -90.16, "model": "vehicle.volkswagen.t2", "color": "73,12,12"},
            {"x": -84.96, "y": -23.28, "z": 0.28, "yaw": 89.84, "model": "vehicle.audi.tt", "color": "16,44,21"},
            {"x": -88.71, "y": -119.57, "z": 0.28, "yaw": 89.84, "model": "vehicle.lincoln.mkz_2017", "color": "16,16,16"},
            {"x": -74.39, "y": -18.55, "z": 0.28, "yaw": -90.16, "model": "vehicle.mercedes.coupe_2020", "color": "73,0,0"},
            {"x": -88.31, "y": 21.53, "z": 0.35, "yaw": 89.84, "model": "vehicle.mini.cooper_s_2021", "color": "215,88,0"},
            {"x": -84.81, "y": 27.02, "z": 0.35, "yaw": 89.84, "model": "vehicle.ford.crown", "color": "255,185,0"},
            {"x": -88.17, "y": 36.74, "z": 0.58, "yaw": 89.79, "model": "vehicle.carlamotors.firetruck", "color": "234,0,0"},
            {"x": -84.77, "y": 51.93, "z": 1.08, "yaw": 89.79, "model": "vehicle.ford.ambulance", "color": "231,231,231"},
            {"x": -88.25, "y": 106.21, "z": 0.72, "yaw": 89.79, "model": "vehicle.nissan.micra", "color": "28,0,46"},
            {"x": -84.65, "y": 114.40, "z": 0.52, "yaw": 89.79, "model": "vehicle.citroen.c3", "color": "217,217,217"},
            {"x": -87.97, "y": 87.14, "z": 0.94, "yaw": 89.79, "model": "vehicle.dodge.charger_2020", "color": "211,142,0"},
            {"x": -84.57, "y": 96.63, "z": 0.74, "yaw": 89.79, "model": "vehicle.audi.tt", "color": "168,0,27"},
            {"x": -88.06, "y": 66.84, "z": 1.14, "yaw": 89.79, "model": "vehicle.nissan.patrol", "color": "183,187,162"},
            {"x": -84.66, "y": 72.43, "z": 1.14, "yaw": 89.79, "model": "vehicle.volkswagen.t2", "color": "105,103,0"},
            {"x": -77.53, "y": 109.83, "z": 0.53, "yaw": -90.16, "model": "vehicle.toyota.prius", "color": "255,0,0"},
            {"x": -74.03, "y": 119.12, "z": 0.43, "yaw": -90.16, "model": "vehicle.toyota.prius", "color": "234,0,0"},
            {"x": -74.39, "y": 77.35, "z": 1.10, "yaw": -90.16, "model": "vehicle.lincoln.mkz_2017", "color": "16,16,16"},
            {"x": -77.89, "y": 87.73, "z": 1.00, "yaw": -90.16, "model": "vehicle.bmw.grandtourer", "color": "0,0,0"},
            {"x": -74.39, "y": 99.72, "z": 0.70, "yaw": -90.16, "model": "vehicle.volkswagen.t2", "color": "73,12,12"},
            {"x": -77.89, "y": 23.34, "z": 0.35, "yaw": -90.16, "model": "vehicle.audi.tt", "color": "16,44,21"},
            {"x": -74.39, "y": 25.73, "z": 0.35, "yaw": -90.16, "model": "vehicle.lincoln.mkz_2017", "color": "16,16,16"},
            {"x": -77.89, "y": 33.21, "z": 0.65, "yaw": -90.16, "model": "vehicle.mercedes.coupe_2020", "color": "73,0,0"},
            {"x": -74.39, "y": 42.00, "z": 0.95, "yaw": -90.16, "model": "vehicle.mini.cooper_s_2021", "color": "215,88,0"},
            {"x": -77.89, "y": 52.04, "z": 1.01, "yaw": -90.16, "model": "vehicle.ford.crown", "color": "255,185,0", "role": "lidar_target"},
            {"x": -74.39, "y": 59.43, "z": 1.21, "yaw": -90.16, "model": "vehicle.ford.ambulance", "color": "231,231,231", "role": "camera_target"},
        ]

        # Cross-checked against a live dump of this map's spawn points
        # (world.get_map().get_spawn_points()): every one of the 28
        # surround_points coordinates is an exact match to a genuine,
        # guaranteed-valid spawn point. So the handful that still fail each
        # run aren't off-road/bad geometry -- most likely several of these
        # points are only ~8-10m apart, and large vehicle models (firetruck,
        # ambulance) spawned nearby block their close neighbors. Rather than
        # chase which exact points survive, use the full spawn point list as
        # a broader pool: fall back to the nearest free one when a preferred
        # point fails, then top up with extra background vehicles at
        # whatever's still free near the scene until TARGET_SURROUND_VEHICLES
        # is reached (or candidates run out).
        TARGET_SURROUND_VEHICLES = 30
        TOPUP_MODELS = [
            "vehicle.audi.tt", "vehicle.bmw.grandtourer", "vehicle.toyota.prius",
            "vehicle.lincoln.mkz_2017", "vehicle.mini.cooper_s_2021",
            "vehicle.mercedes.coupe_2020", "vehicle.nissan.micra", "vehicle.citroen.c3",
        ]
        INTERSECTION_CENTER = carla.Location(x=-82.0, y=-20.0, z=0.0)

        map_spawn_points = world.get_map().get_spawn_points()
        used_fallback_indices = set()

        # Reserve each preferred point's matching map-spawn-point index up
        # front (whether it ends up succeeding or failing), so the fallback/
        # top-up logic never wastes an attempt re-trying the same spot.
        for sp in surround_points:
            for i, mp in enumerate(map_spawn_points):
                if abs(mp.location.x - sp['x']) < 0.1 and abs(mp.location.y - sp['y']) < 0.1:
                    used_fallback_indices.add(i)
                    break

        for sp in surround_points:
            bp = blueprint_library.find(sp['model'])
            if bp.has_attribute('color'):
                bp.set_attribute('color', sp['color'])
            if 'role' in sp:
                bp.set_attribute('role_name', sp['role'])
            transform = carla.Transform(
                carla.Location(x=sp['x'], y=sp['y'], z=sp['z']),
                carla.Rotation(yaw=sp['yaw']))
            veh = _spawn_with_retry(world, bp, transform,
                                     fallback_points=map_spawn_points,
                                     used_fallback_indices=used_fallback_indices)
            if veh:
                veh.set_autopilot(True, tm_port)
                self.surrounding_vehicles.append(veh)
                tm.vehicle_percentage_speed_difference(veh, 70.0)
            else:
                print("[SpawnFail] surrounding vehicle '{}' at ({}, {}, {}) yaw={} -- spot blocked/invalid".format(
                    sp['model'], sp['x'], sp['y'], sp['z'], sp['yaw']))

        if len(self.surrounding_vehicles) < TARGET_SURROUND_VEHICLES:
            remaining = [i for i in range(len(map_spawn_points)) if i not in used_fallback_indices]
            remaining.sort(key=lambda i: INTERSECTION_CENTER.distance(map_spawn_points[i].location))

            for i in remaining:
                if len(self.surrounding_vehicles) >= TARGET_SURROUND_VEHICLES:
                    break
                bp = blueprint_library.find(random.choice(TOPUP_MODELS))
                veh = world.try_spawn_actor(bp, map_spawn_points[i])
                if veh:
                    used_fallback_indices.add(i)
                    veh.set_autopilot(True, tm_port)
                    self.surrounding_vehicles.append(veh)
                    tm.vehicle_percentage_speed_difference(veh, 70.0)

        print("[TopUp] surrounding vehicle count: {} (target {})".format(
            len(self.surrounding_vehicles), TARGET_SURROUND_VEHICLES))

        # world.get_actors() can lag behind a batch of try_spawn_actor() calls
        # that all succeeded without any world.tick() in between -- commit
        # the frame now so list_vehicles() below (and the SensorRecorder's
        # vehicle_index selection) actually sees everyone who was just
        # spawned, not a stale pre-topup snapshot.
        if CarlaDataProvider.is_sync_mode():
            world.tick()
        else:
            world.wait_for_tick()

        # ---------- SENSOR RECORDING: pick ONE vehicle + ONE sensor type ----------
        # Selectable at launch time via environment variables, no code edits needed:
        #   RECORD_VEHICLE_INDEX -> 0-based ordinal into the vehicle list printed
        #                           below (spawn order: ego, then colliding, then
        #                           surrounding vehicles). Out-of-range -> a clear
        #                           alert is printed and recording is disabled,
        #                           the scenario still runs normally.
        #   RECORD_VEHICLE_ROLE_NAME -> e.g. "lidar_target" or "camera_target"
        #   RECORD_SENSOR_TYPE   -> "cameras", "lidar", or "both" (default "cameras")
        #   RECORD_OUTPUT_DIR    -> output root (default "sensor_output")
        #   RECORD_FPS           -> must match world fixed_delta_seconds (default 20.0)
        vehicles = list_vehicles(world)
        print("[SensorRecorder] Vehicles available for recording (0..{}):".format(len(vehicles) - 1))
        for v in vehicles:
            print("    index={:<3d} id={:<4d} role_name='{}' type={}".format(
                v["index"], v["id"], v["role_name"], v["type_id"]))

        record_vehicle_index_env = os.environ.get("RECORD_VEHICLE_INDEX")
        record_vehicle_index = int(record_vehicle_index_env) if record_vehicle_index_env is not None else None
        record_role_name = os.environ.get("RECORD_VEHICLE_ROLE_NAME")
        if record_vehicle_index is None and record_role_name is None:
            record_vehicle_index = 0  # default: record the ego vehicle
        record_sensor_type = os.environ.get("RECORD_SENSOR_TYPE", "cameras")
        record_output_dir = os.environ.get("RECORD_OUTPUT_DIR", "sensor_output")
        record_fps = float(os.environ.get("RECORD_FPS", "20.0"))

        try:
            self.recorder = SensorRecorder(
                world=world,
                output_dir=record_output_dir,
                scenario_name="FiveWaySignalisedIntersectionCollision",
                fps=record_fps,
                sensor_type=record_sensor_type,
                vehicle_index=record_vehicle_index,
                vehicle_role_name=record_role_name,
            )
            self.recorder.setup_sensors()
            print("[SensorRecorder] Recording vehicle_label='{}' sensor_type='{}' -> {}".format(
                self.recorder.vehicle_label, record_sensor_type, self.recorder.output_dir))
        except ValueError as e:
            self.recorder = None
            print("[SensorRecorder] ALERT - recording disabled: {}".format(e))

        # Spectator view
        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            carla.Location(x=-25.74, y=9.18, z=32.49),
            carla.Rotation(pitch=-30.12, yaw=178.78)))

    def _on_collision(self, vehicle):
        if vehicle.id in self._collided_vehicle_ids:
            return  # already handled -- avoid re-triggering every tick while stuck in contact
        self._collided_vehicle_ids.add(vehicle.id)
        print(f"[Collision] Vehicle {vehicle.id} impacted, stopping immediately.")
        vehicle.apply_control(VehicleControl(brake=1.0, hand_brake=True))
        vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))

    def _create_behavior(self):
        # SUCCESS_ON_ONE: the scenario ends as soon as ANY child succeeds --
        # either the ego finishes its drive distance, or (typically sooner)
        # 2 seconds after the colliding vehicles crash into each other.
        root = py_trees.composites.Parallel(
            "Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(KeepVelocity(self.ego_vehicles[0], target_velocity=10.0))
        for idx, cv in enumerate(self.colliding_vehicles):
            root.add_child(ManualThrottle(cv, name=f"ManualThrottle_Collider{idx}"))
        if self.recorder is not None:
            root.add_child(RecordSensorsTick(self.recorder))
        root.add_child(EndAfterCollision(self._collided_vehicle_ids, delay=2.0))
        root.add_child(DriveDistance(self.ego_vehicles[0], distance=150.0))
        return root

    def _create_test_criteria(self):
        return [CollisionTest(self.ego_vehicles[0])]

    def remove_all_actors(self):
        # NOTE: basic_scenario.py never calls a method named "end_scenario" --
        # remove_all_actors() is the actual hook scenario_runner.py invokes
        # once the run finishes (see scenario_runner.py: scenario.remove_all_actors()).
        if self.recorder is not None:
            try:
                self.recorder.on_tick()  # flush the last buffered frame
            except Exception:
                pass
            self.recorder.destroy()
            self.recorder = None

        for sensor in self.collision_sensors:
            try:
                sensor.destroy()
            except RuntimeError:
                pass
        self.collision_sensors = []

        for v in self.surrounding_vehicles + self.colliding_vehicles + self.ego_vehicles:
            try:
                v.destroy()
            except RuntimeError:
                pass

        client = CarlaDataProvider.get_client()
        if client is not None:
            try:
                client.get_trafficmanager().set_synchronous_mode(False)
            except RuntimeError:
                pass
        super(FiveWaySignalisedIntersectionCollision, self).remove_all_actors()
