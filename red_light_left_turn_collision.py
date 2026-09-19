"""
Red light / left-turn collision (Town03).

A vehicle runs a red light straight through a signalised intersection and
collides with another vehicle making a left turn across its path, surrounded
by 17 background traffic vehicles. An ego vehicle is a separate bystander,
not involved in the crash. Sensor data for one chosen vehicle (ego, either
colliding vehicle, or any background vehicle) is recorded via
sensor_recorder.SensorRecorder -- see the repo README for how to pick which
vehicle/sensor type to record.
"""

import py_trees
import carla
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import WaypointFollower, KeepVelocity
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
import os

from sensor_recorder import SensorRecorder, list_vehicles


def EndAfterCollision(collided_ids, delay=1.0, name="EndAfterCollision"):
    """
    py_trees behavior: RUNNING until `collided_ids` (a shared, mutable set
    populated by the two colliding vehicles' collision sensors) becomes
    non-empty, then RUNNING for `delay` more simulated seconds, then
    SUCCESS. Pair with a root Parallel using ParallelPolicy.SUCCESS_ON_ONE so
    the scenario ends as soon as this succeeds -- without needing every
    other perpetually-RUNNING behavior (KeepVelocity, WaypointFollower,
    RecordSensorsTick, ...) to also succeed.
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


class RedLightLeftTurnCollision(BasicScenario):
    """
    Two dedicated actors collide at a signalised intersection: one runs the
    red light straight through, the other is making a left turn across its
    path. The ego vehicle is a separate bystander/observer (not involved in
    the crash) -- same role split as the five-way/four-way/stationary-hazard
    recorded scenarios, so vehicle indices land the same way: 0=ego,
    1=red-light runner, 2=left-turn vehicle, 3.. = background traffic.

    Records synchronised multi-camera + LiDAR data for one chosen vehicle
    (see RECORD_* environment variables below) and ends the scenario a fixed
    delay after the collision is detected.
    """

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True):
        self.timeout = 60
        self.surrounding_vehicles = []
        self.collision_sensors = []
        self.recorder = None
        self.collided_vehicle_ids = set()

        super(RedLightLeftTurnCollision, self).__init__(
            name="RedLightLeftTurnCollision",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            criteria_enable=criteria_enable)

    def _setup_scenario_trigger(self, config):
        # scenario_parser.py auto-populates config.trigger_points with the
        # ego's OWN spawn transform. BasicScenario's default
        # _setup_scenario_trigger() would then return
        # InTimeToArrivalToLocation(ego, 2.0, ego's own spawn point) as the
        # first child of the outer behavior Sequence -- a condition that
        # needs the ego to already be moving toward a point it's currently
        # sitting motionless on, so it never succeeds and _create_behavior()'s
        # tree never ticks at all. Same fix as the other recorded scenarios.
        return None

    def _initialize_actors(self, config):
        if hasattr(config, "weather") and config.weather is not None:
            CarlaDataProvider.get_world().set_weather(config.weather)

        world = CarlaDataProvider.get_world()
        blueprint_library = world.get_blueprint_library()
        tm = CarlaDataProvider.get_client().get_trafficmanager()
        tm_port = tm.get_port()

        # ---------- CLEANUP: self-heal from a previous interrupted/crashed run ----------
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
        # settling; spawning immediately can spuriously report a perfectly
        # valid spawn point as blocked. Same fix five-way/four-way needed.
        settle_ticks = 10
        for _ in range(settle_ticks):
            if CarlaDataProvider.is_sync_mode():
                world.tick()
            else:
                world.wait_for_tick()

        # ---------- EGO: bystander/observer, not involved in the crash ----------
        ego_vehicle = self.ego_vehicles[0]
        ego_vehicle.set_autopilot(False)
        ego_vehicle.set_simulate_physics(True)

        # ---------- COLLIDING VEHICLES (both <other_actor> entries, spawned
        # right after ego so they land at low, stable indices 1 and 2) ----------
        # config.other_actors[0] must be the red-light runner and [1] the
        # left-turn vehicle -- that's just document order in the XML.
        for actor in config.other_actors:
            other = CarlaDataProvider.request_new_actor(
                actor.model, actor.transform, actor.rolename, color=actor.color)
            retry = 0
            while other is None and retry < 5:
                # Spurious "blocked" report right after reload -- tick once
                # more and retry rather than giving up immediately.
                if CarlaDataProvider.is_sync_mode():
                    world.tick()
                else:
                    world.wait_for_tick()
                other = CarlaDataProvider.request_new_actor(
                    actor.model, actor.transform, actor.rolename, color=actor.color)
                retry += 1
            if other is None:
                print(f"[SpawnFail] other actor '{actor.model}' (role='{actor.rolename}') "
                      f"at {actor.transform.location} -- spot blocked/invalid after {retry} retries")
                continue
            self.other_actors.append(other)
            other.set_simulate_physics(True)
            other.set_autopilot(False)

            col_bp = blueprint_library.find('sensor.other.collision')
            col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=other)
            col_sensor.listen(lambda event, v=other: self._on_collision(v))
            self.collision_sensors.append(col_sensor)

            print(f"[Manual] Spawned other actor: {actor.model} (role='{actor.rolename}') at {actor.transform.location}")

        # ---------- Background traffic (autopilot, for visual context) ----------
        spawn_points = [
            {"x": 242.70, "y": 129.08, "z": 1.35, "yaw": -88.61, "model": "vehicle.toyota.prius", "color": "234,0,0"},
            {"x": 239.20, "y": 115.79, "z": 0.95, "yaw": -88.61, "model": "vehicle.lincoln.mkz_2017", "color": "16,16,16"},
            {"x": 243.12, "y": 103.16, "z": 0.97, "yaw": -88.61, "model": "vehicle.bmw.grandtourer", "color": "0,0,0"},
            {"x": 245.19, "y": 28.10, "z": 0.28, "yaw": -88.61, "model": "vehicle.volkswagen.t2", "color": "73,12,12"},
            {"x": 129.13, "y": 5.37, "z": 0.28, "yaw": 0.86, "model": "vehicle.audi.tt", "color": "16,44,21"},
            {"x": 220.41, "y": -5.09, "z": 0.28, "yaw": -179.14, "model": "vehicle.lincoln.mkz_2017", "color": "16,16,16"},
            {"x": 227.26, "y": -1.59, "z": 0.28, "yaw": -179.14, "model": "vehicle.mercedes.coupe_2020", "color": "73,0,0"},
            {"x": 204.70, "y": 58.80, "z": 0.28, "yaw": 179.85, "model": "vehicle.mini.cooper_s_2021", "color": "215,88,0"},
            {"x": 230.82, "y": 35.71, "z": 0.28, "yaw": 91.39, "model": "vehicle.ford.crown", "color": "255,185,0"},
            {"x": 192.40, "y": 9.81, "z": 0.28, "yaw": 0.86, "model": "vehicle.carlamotors.firetruck", "color": "234,0,0"},
            {"x": 131.04, "y": 62.49, "z": 0.28, "yaw": -0.15, "model": "vehicle.ford.ambulance", "color": "231,231,231"},
            {"x": 124.70, "y": 59.01, "z": 0.28, "yaw": 179.85, "model": "vehicle.nissan.micra", "color": "28,0,46"},
            {"x": 120.08, "y": 8.87, "z": 0.28, "yaw": 0.86, "model": "vehicle.citroen.c3", "color": "217,217,217"},
            {"x": 140.54, "y": 8.87, "z": 0.28, "yaw": 0.86, "model": "vehicle.audi.tt", "color": "168,0,27"},
            {"x": 149.59, "y": 5.37, "z": 0.28, "yaw": 0.86, "model": "vehicle.nissan.patrol", "color": "183,187,162"},
            {"x": 65.52, "y": 7.81, "z": 0.28, "yaw": 0.86, "model": "vehicle.volkswagen.t2", "color": "105,103,0"},
            {"x": 76.47, "y": 4.31, "z": 0.28, "yaw": 0.86, "model": "vehicle.toyota.prius", "color": "255,0,0"},
        ]

        for sp in spawn_points:
            vehicle_bp = blueprint_library.find(sp["model"])
            if vehicle_bp.has_attribute('color'):
                vehicle_bp.set_attribute('color', sp["color"])
            # Deliberately NOT setting role_name here: SensorRecorder uses
            # role_name (if non-empty) as the output folder name
            # (vehicle_label). If every background vehicle shared the same
            # role_name, recording different indices across a sweep would
            # keep overwriting the same output folder instead of each
            # getting its own -- leaving it unset falls back to a unique
            # "vehicle_<id>" label per vehicle.

            transform = carla.Transform(
                carla.Location(x=sp["x"], y=sp["y"], z=sp["z"]),
                carla.Rotation(yaw=sp["yaw"])
            )
            vehicle = world.try_spawn_actor(vehicle_bp, transform)
            if vehicle:
                vehicle.set_autopilot(True, tm_port)
                self.surrounding_vehicles.append(vehicle)
                tm.vehicle_percentage_speed_difference(vehicle, 70)

        print("[TopUp] surrounding vehicle count: {}".format(len(self.surrounding_vehicles)))

        # world.get_actors() can lag behind a batch of try_spawn_actor() calls
        # with no world.tick() in between -- commit the frame now so
        # list_vehicles() below sees everyone who was just spawned.
        if CarlaDataProvider.is_sync_mode():
            world.tick()
        else:
            world.wait_for_tick()

        # ---------- SENSOR RECORDING: pick ONE vehicle + ONE sensor type ----------
        # Selectable at launch time via environment variables, no code edits needed:
        #   RECORD_VEHICLE_INDEX -> 0-based ordinal into the vehicle list printed
        #                           at scenario start (spawn order: ego=0,
        #                           red-light runner=1, left-turn vehicle=2,
        #                           then background traffic). Default 0 (ego).
        #   RECORD_VEHICLE_ROLE_NAME -> select by role name instead of index
        #                           (ego='hero', 'red_light_vehicle', or
        #                           'left_turn_vehicle')
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
                scenario_name="RedLightLeftTurnCollision",
                fps=record_fps,
                sensor_type=record_sensor_type,
                vehicle_index=record_vehicle_index,
                vehicle_role_name=record_role_name,
            )
            self.recorder.setup_sensors()
            print("[SensorRecorder] Recording vehicle_label='{}' sensor_type='{}' -> {}".format(
                self.recorder.vehicle_label, record_sensor_type, self.recorder.output_dir))
        except Exception as e:
            self.recorder = None
            print("[SensorRecorder] ALERT - recording disabled: {}: {}".format(type(e).__name__, e))

        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            carla.Location(x=269.81, y=28.48, z=19.87),
            carla.Rotation(pitch=-13.53, yaw=154.74, roll=0.00)
        ))

    def _on_collision(self, vehicle):
        if vehicle.id in self.collided_vehicle_ids:
            return  # already handled -- avoid re-triggering every tick while stuck in contact
        self.collided_vehicle_ids.add(vehicle.id)
        print(f"[Collision] Vehicle {vehicle.id} impacted -- ending scenario in 1s.")
        control = carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
        vehicle.apply_control(control)
        vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))

    def _create_behavior(self):
        # SUCCESS_ON_ONE (not SUCCESS_ON_ALL): KeepVelocity, the
        # WaypointFollowers, and RecordSensorsTick all run forever and never
        # succeed on their own, so the scenario has to end as soon as ANY
        # one child succeeds -- either EndAfterCollision (1s after the two
        # colliders hit each other) or the ego completing its fallback drive
        # distance.
        root = py_trees.composites.Parallel("Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        ego_vehicle = self.ego_vehicles[0]
        root.add_child(KeepVelocity(ego_vehicle, target_velocity=10.0, name="EgoKeepVelocity"))

        # other_actors[0] = red-light runner, [1] = left-turn vehicle.
        # avoid_collision=False is what makes them ignore the light/each
        # other and actually crash, rather than braking to avoid it. The
        # red-light runner drives much faster so it actually reaches the
        # intersection in time to hit the (slower) left-turning vehicle,
        # instead of the left-turner clearing the crossing first.
        # NOTE: hardcoded here rather than read from actor.speed/config --
        # ActorConfigurationData.parse_from_node() never casts the XML
        # "speed" attribute to a number (it stays a raw string, or the int
        # literal 0 if absent), so target_speed = actor.speed / 3.6 would
        # throw a TypeError the moment a "speed" attribute is ever added to
        # the XML.
        ROLE_SPEED_KMPH = {
            "red_light_vehicle": 90.0,
            "left_turn_vehicle": 20.0,
        }
        DEFAULT_COLLIDER_SPEED_KMPH = 23.0
        for i, actor in enumerate(self.other_actors):
            # self.other_actors holds carla.Actor objects (returned by
            # request_new_actor), not ActorConfigurationData -- role name
            # lives in .attributes, there's no .rolename here.
            role_name = actor.attributes.get('role_name', '')
            actor_speed_kmph = ROLE_SPEED_KMPH.get(role_name, DEFAULT_COLLIDER_SPEED_KMPH)
            root.add_child(WaypointFollower(
                actor=actor,
                target_speed=actor_speed_kmph / 3.6,
                avoid_collision=False,
                name=f"WaypointFollower_Collider{i}"))

        if self.recorder is not None:
            root.add_child(RecordSensorsTick(self.recorder))

        root.add_child(EndAfterCollision(self.collided_vehicle_ids, delay=1.0))
        root.add_child(DriveDistance(ego_vehicle, distance=150.0, name="ScenarioCompletion"))
        return root

    def _create_test_criteria(self):
        return [CollisionTest(self.ego_vehicles[0])]

    def remove_all_actors(self):
        # basic_scenario.py never calls a method named "end_scenario" --
        # remove_all_actors() is the actual hook scenario_runner.py invokes
        # once the run finishes.
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

        client = CarlaDataProvider.get_client()
        for v in self.surrounding_vehicles + self.other_actors + self.ego_vehicles:
            try:
                v.destroy()
            except RuntimeError:
                pass
        if client is not None:
            try:
                client.get_trafficmanager().set_synchronous_mode(False)
            except RuntimeError:
                pass

        super(RedLightLeftTurnCollision, self).remove_all_actors()
