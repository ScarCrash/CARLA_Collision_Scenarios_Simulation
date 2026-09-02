"""
4-way unsignalised intersection collision (Town04).

Three vehicles collide near a 4-way unsignalised intersection, surrounded by
~30 background traffic vehicles. An ego vehicle drives through and observes.
Sensor data for one chosen vehicle (ego, a colliding vehicle, or any
background vehicle) is recorded via sensor_recorder.SensorRecorder -- see
the repo README for how to pick which vehicle/sensor type to record.
"""

import os
import random
from math import radians, cos, sin

import py_trees
import carla

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import KeepVelocity
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from sensor_recorder import SensorRecorder, list_vehicles


def _in_collider_path(x, y, colliders_config, lateral_thresh=6.0, dist_thresh=60.0):
    """
    True if (x, y) sits ahead of any colliding vehicle's start position,
    within lateral_thresh of its lane and dist_thresh of its path length.
    Used to keep fallback/top-up spawns out of every collider's way, not
    just the hand-curated point lists.
    """
    for cp in colliders_config:
        fx, fy = cos(radians(cp['yaw'])), sin(radians(cp['yaw']))
        dx, dy = x - cp['x'], y - cp['y']
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 0.01:
            continue
        along = dx * fx + dy * fy
        lateral = abs(dx * (-fy) + dy * fx)
        if along > 0 and lateral < lateral_thresh and dist < dist_thresh:
            return True
    return False


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
    the nudge lands on a different lane/road than the original point was
    on -- that's why fallback is tried first whenever it's available.
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
    (KeepVelocity, ForceStraightVelocity, ...) to also reach SUCCESS.
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


def ForceStraightVelocity(vehicle, velocity, stopped_ids=None, name="ForceStraightVelocity"):
    """
    py_trees behavior: reasserts a fixed velocity + zero angular velocity on
    `vehicle` every tick. set_target_velocity() is a one-shot kinematic
    override -- without continuous reassertion, friction/drag or a collision
    impulse can slow or turn the vehicle over time. Re-applying it every
    tick forces it to hold a straight line at constant speed regardless.

    If `stopped_ids` is given (a shared, mutable set), this behavior stops
    reasserting velocity once `vehicle.id` appears in it -- pair with a
    collision sensor that adds the id on impact to make a vehicle hold speed
    right up until it hits something, then let physics take over instead of
    forcing it to keep moving through the collision.
    """
    class _ForceStraight(py_trees.behaviour.Behaviour):
        def __init__(self):
            super(_ForceStraight, self).__init__(name)
            self._vehicle = vehicle
            self._velocity = velocity
            self._stopped_ids = stopped_ids

        def update(self):
            if self._stopped_ids is not None and self._vehicle.id in self._stopped_ids:
                return py_trees.common.Status.RUNNING
            self._vehicle.set_target_velocity(self._velocity)
            self._vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            return py_trees.common.Status.RUNNING

    return _ForceStraight()


class FourWayUnsignalisedIntersectionCollision(BasicScenario):
    """
    Vehicle order (spawn order == vehicle index):
      N=0            ego
      N=1, 2, 3      the 3 colliding vehicles (red sedan, green mini, blue truck)
      N=4 .. N=33    ~19 curated background vehicles + top-up to 30 total
    """

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True):
        self.timeout = 120
        self.surrounding_vehicles = []
        self.colliding_vehicles = []
        self.collision_sensors = []
        self.colliding_points_config = []
        self.late_surrounding_vehicles_config = []
        self.recorder = None
        self.collider_velocity_holds = []  # list of (vehicle, target_velocity) reasserted every tick
        self.stopped_collider_ids = set()  # collider vehicle ids that should stop holding velocity after impact

        super(FourWayUnsignalisedIntersectionCollision, self).__init__(
            name="FourWayUnsignalisedIntersectionCollision",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            criteria_enable=criteria_enable
        )

    def _setup_scenario_trigger(self, config):
        # BasicScenario's default trigger is InTimeToArrivalToLocation(ego,
        # 2.0, ego's own spawn point). Its update() falls back to
        # time_to_arrival=inf whenever the ego's velocity is ~0 -- and the
        # ego starts at velocity 0, only gaining speed from KeepVelocity
        # *inside* the very behavior tree this trigger gates. That's a
        # deadlock: it never succeeds, so _create_behavior()'s whole tree
        # (including the sensor recorder) never ticks. Skip it.
        return None

    def _initialize_actors(self, config):
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
        for _ in range(10):
            if CarlaDataProvider.is_sync_mode():
                world.tick()
            else:
                world.wait_for_tick()

        self.colliding_points_config = [
            {"x": 199.53, "y": -210.40, "z": 0.78, "yaw": 91.31, "speed": 1.8, "model": "vehicle.tesla.model3", "color": "255,0,0"},
            {"x": 105.29, "y": -170.06, "z": 0.60, "yaw": 0.33, "speed": 3.9, "model": "vehicle.mini.cooper_s", "color": "0,255,0"},
            {"x": 227.07, "y": -172.86, "z": 0.56, "yaw": -179.67, "speed": 1.75, "model": "vehicle.carlamotors.carlacola", "color": "0,0,255"},
        ]

        # Config for vehicles near the collision area. Checked each point
        # against every colliding vehicle's forward direction (projection
        # onto heading + lateral offset from its lane) so nothing spawns
        # directly in a collider's way; see README for the full derivation.
        initial_surround_points = [
            {"x": 104.94, "y": -173.56, "z": 0.60, "yaw": -179.67, "model": "vehicle.toyota.prius", "color": "234,0,0"},
            {"x": 131.18, "y": -180.55, "z": 0.40, "yaw": -87.54, "model": "vehicle.lincoln.mkz_2017", "color": "16,16,16"},
            {"x": 201.20, "y": -283.47, "z": 0.48, "yaw": 91.31, "model": "vehicle.audi.tt", "color": "16,44,21"},
            {"x": 284.55, "y": -172.54, "z": 0.54, "yaw": -179.67, "model": "vehicle.lincoln.mkz_2017", "color": "16,16,16", "role": "lidar_target"},
            {"x": 284.78, "y": -169.04, "z": 0.54, "yaw": 0.33, "model": "vehicle.mercedes.coupe_2020", "color": "73,0,0"},
            {"x": 285.91, "y": -121.88, "z": 0.58, "yaw": -179.08, "model": "vehicle.mini.cooper_s_2021", "color": "0,0,0"},
            {"x": 314.29, "y": -146.64, "z": 0.28, "yaw": -89.49, "model": "vehicle.carlamotors.firetruck", "color": "234,0,0"},
            {"x": 233.21, "y": -307.65, "z": 0.38, "yaw": 0.59, "model": "vehicle.nissan.micra", "color": "28,0,46"},
            {"x": 226.96, "y": -311.22, "z": 0.38, "yaw": -179.41, "model": "vehicle.citroen.c3", "color": "217,217,217"},
            {"x": 255.27, "y": -280.21, "z": 0.58, "yaw": 90.18, "model": "vehicle.dodge.charger_2020", "color": "211,142,0"},
            {"x": 229.87, "y": -246.03, "z": 0.28, "yaw": -0.39, "model": "vehicle.audi.tt", "color": "168,0,27"},
            {"x": 37.25, "y": -170.44, "z": 0.60, "yaw": 0.33, "model": "vehicle.ford.crown", "color": "255,185,0"},
            {"x": 83.19, "y": -170.19, "z": 0.60, "yaw": 0.33, "model": "vehicle.ford.ambulance", "color": "231,231,231"},
            {"x": 246.87, "y": -172.75, "z": 0.60, "yaw": -179.67, "model": "vehicle.bmw.grandtourer", "color": "0,0,0"},
            {"x": 203.30, "y": -222.19, "z": 0.78, "yaw": -88.69, "model": "vehicle.volkswagen.t2", "color": "73,12,12"},
            {"x": 199.99, "y": -230.69, "z": 0.78, "yaw": 91.31, "speed": 2.2, "model": "vehicle.tesla.model3", "color": "255,0,0"},
        ]

        # Late-spawning vehicles: every one of these duplicates the exact
        # coordinates of an already-placed colliding/initial vehicle above,
        # so the primary attempt always fails and falls back to a nearby
        # free spot instead (handled by _spawn_with_retry's fallback).
        self.late_surrounding_vehicles_config = [
            {"x": 246.87, "y": -172.75, "z": 0.60, "yaw": -179.67, "model": "vehicle.bmw.grandtourer", "color": "0,0,0"},
            {"x": 203.30, "y": -222.19, "z": 0.78, "yaw": -88.69, "model": "vehicle.volkswagen.t2", "color": "73,12,12"},
            {"x": 201.20, "y": -283.47, "z": 0.48, "yaw": 91.31, "model": "vehicle.audi.tt", "color": "16,44,21", "role": "camera_target"},
        ]

        TARGET_SURROUND_VEHICLES = 30
        TOPUP_MODELS = [
            "vehicle.audi.tt", "vehicle.bmw.grandtourer", "vehicle.toyota.prius",
            "vehicle.lincoln.mkz_2017", "vehicle.mini.cooper_s_2021",
            "vehicle.mercedes.coupe_2020", "vehicle.nissan.micra", "vehicle.citroen.c3",
        ]
        INTERSECTION_CENTER = carla.Location(x=200.0, y=-220.0, z=0.0)

        map_spawn_points = world.get_map().get_spawn_points()
        used_fallback_indices = set()

        # Reserve every preferred point's matching map-spawn-point index up
        # front (colliding + both surrounding lists), so fallback/top-up
        # never wastes an attempt re-trying an already-claimed spot.
        all_preferred_points = (self.colliding_points_config + initial_surround_points
                                 + self.late_surrounding_vehicles_config)
        for sp in all_preferred_points:
            for i, mp in enumerate(map_spawn_points):
                if abs(mp.location.x - sp['x']) < 0.1 and abs(mp.location.y - sp['y']) < 0.1:
                    used_fallback_indices.add(i)
                    break

        # Also reserve (i.e. exclude) every map spawn point that sits ahead
        # of any colliding vehicle's path -- otherwise fallback rerouting or
        # the top-up loop below can freely place a vehicle back in a
        # collider's way.
        for i, mp in enumerate(map_spawn_points):
            if i in used_fallback_indices:
                continue
            if _in_collider_path(mp.location.x, mp.location.y, self.colliding_points_config):
                used_fallback_indices.add(i)

        # Colliding vehicles spawn first so they land at low vehicle indices
        # (N=1, N=2, ... right after the ego at N=0) -- easy to pick out from
        # the surrounding traffic that follows.
        self.spawn_colliding_vehicles()

        print("Spawning initial vehicles...")
        for sp in initial_surround_points:
            bp = blueprint_library.find(sp['model'])
            if bp.has_attribute('color'):
                bp.set_attribute('color', sp['color'])
            if 'role' in sp:
                bp.set_attribute('role_name', sp['role'])

            transform = carla.Transform(carla.Location(x=sp['x'], y=sp['y'], z=sp['z']),
                                         carla.Rotation(yaw=sp['yaw']))
            veh = _spawn_with_retry(world, bp, transform,
                                     fallback_points=map_spawn_points,
                                     used_fallback_indices=used_fallback_indices)
            if veh:
                veh.set_autopilot(True, tm_port)
                tm.vehicle_percentage_speed_difference(veh, 80.0)
                self.surrounding_vehicles.append(veh)
            else:
                print("[SpawnFail] surrounding vehicle '{}' at ({}, {}, {}) yaw={} -- spot blocked/invalid".format(
                    sp['model'], sp['x'], sp['y'], sp['z'], sp['yaw']))

        self.spawn_late_surrounding_vehicles(fallback_points=map_spawn_points,
                                              used_fallback_indices=used_fallback_indices)

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
                    tm.vehicle_percentage_speed_difference(veh, 80.0)
                    self.surrounding_vehicles.append(veh)

        print("[TopUp] surrounding vehicle count: {} (target {})".format(
            len(self.surrounding_vehicles), TARGET_SURROUND_VEHICLES))

        # world.get_actors() can lag behind a batch of try_spawn_actor() calls
        # that all succeeded without any world.tick() in between -- commit
        # the frame now so list_vehicles() below actually sees everyone who
        # was just spawned, not a stale pre-spawn snapshot.
        if CarlaDataProvider.is_sync_mode():
            world.tick()
        else:
            world.wait_for_tick()

        # ---------- SENSOR RECORDING: pick ONE vehicle + ONE sensor type ----------
        # Selectable at launch time via environment variables, no code edits needed:
        #   RECORD_VEHICLE_INDEX -> 0-based ordinal into the vehicle list printed
        #                           below (spawn order: ego, then colliding,
        #                           then surrounding, then late/top-up).
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
                scenario_name="FourWayUnsignalisedIntersectionCollision",
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

        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            carla.Location(x=199.28, y=-185.79, z=57.08),
            carla.Rotation(pitch=-72.97, yaw=89.81, roll=0.00)
        ))

    def spawn_colliding_vehicles(self):
        print("Spawning colliding vehicles...")
        world = CarlaDataProvider.get_world()
        blueprint_library = world.get_blueprint_library()
        tm_port = CarlaDataProvider.get_client().get_trafficmanager().get_port()

        for cp in self.colliding_points_config:
            bp = blueprint_library.find(cp['model'])
            if bp.has_attribute('color'):
                bp.set_attribute('color', cp['color'])
            transform = carla.Transform(
                carla.Location(x=cp['x'], y=cp['y'], z=cp['z']),
                carla.Rotation(yaw=cp['yaw'])
            )
            veh = _spawn_with_retry(world, bp, transform)
            if veh:
                veh.set_simulate_physics(True)
                veh.set_autopilot(False, tm_port)
                forward = transform.rotation.get_forward_vector()
                speed = cp.get('speed', 3.0)
                vel_vec = carla.Vector3D(forward.x * speed, forward.y * speed, forward.z * speed)
                veh.set_target_velocity(vel_vec)
                self.colliding_vehicles.append(veh)
                # set_target_velocity() is a one-shot kinematic push -- without
                # continuous reassertion, friction/drag bleeds speed off over
                # time. Hold every colliding vehicle at its intended velocity
                # every tick so they all approach the crash at full speed.
                self.collider_velocity_holds.append((veh, vel_vec))

                if cp.get('color') in ('0,255,0', '255,0,0'):
                    # Red sedan and green mini: stop holding velocity and
                    # brake once they hit something -- or get hit -- instead
                    # of continuing to push through the collision.
                    col_bp = blueprint_library.find('sensor.other.collision')
                    col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=veh)
                    col_sensor.listen(lambda event, v=veh: self._on_collider_collision(v))
                    self.collision_sensors.append(col_sensor)
            else:
                print("[SpawnFail] colliding vehicle '{}' at ({}, {}, {}) yaw={} -- spot blocked/invalid".format(
                    cp['model'], cp['x'], cp['y'], cp['z'], cp['yaw']))

    def _on_collider_collision(self, vehicle):
        if vehicle.id in self.stopped_collider_ids:
            return  # already handled -- avoid re-triggering every tick while stuck in contact
        self.stopped_collider_ids.add(vehicle.id)
        print(f"[Collision] Collider {vehicle.id} impacted, stopping.")
        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))

    def spawn_late_surrounding_vehicles(self, fallback_points=None, used_fallback_indices=None):
        print("Spawning late surrounding vehicles...")
        world = CarlaDataProvider.get_world()
        blueprint_library = world.get_blueprint_library()
        tm = CarlaDataProvider.get_client().get_trafficmanager()
        tm_port = tm.get_port()

        for sp in self.late_surrounding_vehicles_config:
            bp = blueprint_library.find(sp['model'])
            if bp.has_attribute('color'):
                bp.set_attribute('color', sp['color'])
            if 'role' in sp:
                bp.set_attribute('role_name', sp['role'])

            transform = carla.Transform(
                carla.Location(x=sp['x'], y=sp['y'], z=sp['z']),
                carla.Rotation(yaw=sp['yaw'])
            )
            veh = _spawn_with_retry(world, bp, transform,
                                     fallback_points=fallback_points,
                                     used_fallback_indices=used_fallback_indices)
            if veh:
                veh.set_autopilot(True, tm_port)
                tm.vehicle_percentage_speed_difference(veh, 80.0)
                self.surrounding_vehicles.append(veh)
                print(f"Spawned late vehicle: {sp['model']}")
            else:
                print("[SpawnFail] late surrounding vehicle '{}' at ({}, {}, {}) yaw={} -- spot blocked/invalid".format(
                    sp['model'], sp['x'], sp['y'], sp['z'], sp['yaw']))

    def _create_behavior(self):
        # SUCCESS_ON_ONE: the scenario ends as soon as ANY child succeeds --
        # either the ego finishes its drive distance, or (typically sooner)
        # 2 seconds after a colliding vehicle crashes.
        root = py_trees.composites.Parallel("Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(KeepVelocity(self.ego_vehicles[0], target_velocity=10.0, name="EgoKeepVelocity"))
        for idx, (veh, vel) in enumerate(self.collider_velocity_holds):
            root.add_child(ForceStraightVelocity(veh, vel, stopped_ids=self.stopped_collider_ids,
                                                  name="ColliderForceStraight{}".format(idx)))
        if self.recorder is not None:
            root.add_child(RecordSensorsTick(self.recorder))
        root.add_child(EndAfterCollision(self.stopped_collider_ids, delay=2.0))
        root.add_child(DriveDistance(self.ego_vehicles[0], distance=500.0, name="ScenarioCompletion"))
        return root

    def _create_test_criteria(self):
        return [CollisionTest(self.ego_vehicles[0])]

    def remove_all_actors(self):
        # basic_scenario.py never calls a method named "end_scenario" or
        # relies on __del__ timing -- remove_all_actors() is the actual hook
        # scenario_runner.py invokes once the run finishes.
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
        super(FourWayUnsignalisedIntersectionCollision, self).remove_all_actors()

    def __del__(self):
        """
        Cleanup safety net -- remove_all_actors() above is the real hook the
        framework calls; this just covers interpreter-exit edge cases.
        """
        try:
            self.remove_all_actors()
        except Exception:
            pass
