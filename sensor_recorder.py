"""
sensor_recorder.py

Records raw sensor data + calibration for a single chosen vehicle in a CARLA
scenario_runner scenario. No labels / bounding boxes / classification output
-- sensor data only.

Usage pattern:
    1. List the vehicles present in the world and pick one.
    2. Construct a SensorRecorder for that vehicle and a sensor_type
       ('cameras' for the 6-camera surround rig, 'lidar' for the single
       top LiDAR, or 'both' for cameras + lidar together).
    3. Call setup_sensors() once, after the target vehicle exists.
    4. Call on_tick() once per world.tick(), from inside the existing
       scenario tick loop (do not add a second tick() call).
    5. Call destroy() when the scenario ends or on cleanup/exception.

Output layout: <output_dir>/<scenario_name>/<vehicle_label>/
    Camera_Front/, Camera_FrontLeft/, ... (if cameras active)
    lidar01/ (if lidar active)
    calib/  -- one combined XXXXXX.pkl per written frame, always containing
               ego_to_world, all 6 intrinsic_Camera_*, all 6
               lidar_to_Camera_*, and lidar_to_ego (DeepAccident format)
"""

import math
import os
import pickle

import numpy as np
from PIL import Image

import carla


# ---------------------------------------------------------------------------
# Static sensor configuration -- edit these to change camera/LiDAR mounts
# ---------------------------------------------------------------------------

IMAGE_SIZE_X = 1600
IMAGE_SIZE_Y = 900

CAMERA_CONFIGS = [
    # name,               x,     y,    z,   yaw,   fov
    ("Camera_Front",       1.5,  0.0,  1.6,    0,   70),
    ("Camera_FrontLeft",   1.2, -0.5,  1.6,  -55,   70),
    ("Camera_FrontRight",  1.2,  0.5,  1.6,   55,   70),
    ("Camera_Back",       -1.8,  0.0,  1.6,  180,  110),
    ("Camera_BackLeft",   -0.5, -0.5,  1.6, -110,   70),
    ("Camera_BackRight",  -0.5,  0.5,  1.6,  110,   70),
]

# CAMERA_CONFIGS' x/z offsets are tuned for a typical sedan (deliberately
# INSET from the body -- e.g. 1.5m forward sits near the windshield/cabin,
# well short of a sedan's own ~2.4m half-length, which is normal for a
# realistic AV camera mount). A much larger vehicle (e.g. a fire truck) has
# an opaque hood/box extending well past where that inset position would be,
# so the fixed offset ends up buried inside its body instead of near a
# glassy cabin area.
#
# A ratio-based rescale depends on guessing a "typical sedan" reference size
# -- if that guess is off, the result is still short of actually clearing
# the body. Instead, clamp directly to the vehicle's OWN measured front/back
# distance (plus a clearance margin): this is geometrically guaranteed to
# sit outside the body regardless of vehicle proportions, not just "probably
# far enough". bounding_box.location is itself offset from the vehicle's
# origin for many trucks (e.g. a long rear cargo box shifts the box's center
# backward), so front and back distances are computed and clamped
# independently rather than assuming symmetry.
#
# Only clamp for vehicles CLEARLY bigger than a sedan (gated by the three
# LARGE_VEHICLE_EXTENT_*_THRESHOLD constants below, one per axis) -- a no-op
# for every sedan/hatchback-sized vehicle already used and calibrated in the
# other recorded scenarios.
#
# Gated on length OR width OR height, not just length: a fire truck/ambulance
# is caught by the length check, but a boxy cab-over van/bus (e.g.
# vehicle.volkswagen.t2) can be unremarkable in length yet tall AND wide
# enough that the fixed z=1.6 mount and inset x/y offsets still land inside
# its cabin/cargo box -- confirmed visually in the red-light-left-turn
# scenario's v06/v18 (both vehicle.volkswagen.t2): the Front camera shows the
# dashboard/cowl in frame, and the Back camera shows both the roofline AND
# the rear deck, i.e. it's sitting well inside the box rather than near the
# glass.
#
# The width/height thresholds are deliberately set LOW (closer to a typical
# sedan's own real half-width/-height than a generous margin above it) --
# without a live CARLA connection to query each blueprint's exact
# bounding_box, there's no way to pin these exactly, and the two failure
# modes are not symmetric: a vehicle that trips this gate unnecessarily just
# gets pushed a bit further from a "dashboard cam" look toward a "bumper cam"
# look (harmless), while a vehicle that SHOULD trip it but doesn't produces
# the exact occlusion bug this exists to prevent. Erring toward triggering
# more often is the safe direction.
LARGE_VEHICLE_EXTENT_X_THRESHOLD = 2.6
LARGE_VEHICLE_EXTENT_Y_THRESHOLD = 0.95
LARGE_VEHICLE_EXTENT_Z_THRESHOLD = 0.95
CAMERA_CLEARANCE_MARGIN = 0.4


def _camera_mount_offset(vehicle, x, y, z):
    bbox = vehicle.bounding_box
    is_large = (bbox.extent.x > LARGE_VEHICLE_EXTENT_X_THRESHOLD
                or bbox.extent.y > LARGE_VEHICLE_EXTENT_Y_THRESHOLD
                or bbox.extent.z > LARGE_VEHICLE_EXTENT_Z_THRESHOLD)
    if not is_large:
        return x, y, z

    front_dist = bbox.location.x + bbox.extent.x  # origin -> front bumper
    back_dist = bbox.extent.x - bbox.location.x   # origin -> rear bumper
    right_dist = bbox.location.y + bbox.extent.y  # origin -> right side
    left_dist = bbox.extent.y - bbox.location.y   # origin -> left side

    if x >= 0:
        x = max(x, front_dist + CAMERA_CLEARANCE_MARGIN)
    else:
        x = min(x, -(back_dist + CAMERA_CLEARANCE_MARGIN))
    if y >= 0:
        y = max(y, right_dist + CAMERA_CLEARANCE_MARGIN)
    else:
        y = min(y, -(left_dist + CAMERA_CLEARANCE_MARGIN))
    if z > 0:
        top_dist = bbox.location.z + bbox.extent.z  # origin -> roof
        z = max(z, top_dist + CAMERA_CLEARANCE_MARGIN)
    return x, y, z


LIDAR_LOCATION = (0.0, 0.0, 2.0)
LIDAR_YAW = 0
LIDAR_CHANNELS = 32
LIDAR_RANGE = 100
LIDAR_UPPER_FOV = 10
LIDAR_LOWER_FOV = -30
LIDAR_POINTS_PER_ROTATION = 37500  # rays cast per full rotation, at any fps

SENSOR_TYPES = ("cameras", "lidar", "both")


def _sorted_vehicle_actors(world):
    # Sorted by actor id ascending, which matches spawn order -> gives a
    # stable, deterministic 0..n-1 indexing for a given scenario config.
    return sorted(world.get_actors().filter("vehicle.*"), key=lambda a: a.id)


def list_vehicles(world):
    """
    Enumerate every vehicle actor currently in the world, in spawn order.

    Returns a list of dicts: {"index": int, "id": int, "type_id": str,
    "role_name": str}. "index" is the 0-based ordinal to pass as
    SensorRecorder(vehicle_index=...) -- the simplest way to pick a vehicle.
    """
    vehicles = []
    for i, actor in enumerate(_sorted_vehicle_actors(world)):
        vehicles.append({
            "index": i,
            "id": actor.id,
            "type_id": actor.type_id,
            "role_name": actor.attributes.get("role_name", ""),
        })
    return vehicles


def _resolve_vehicle(world, vehicle_id, vehicle_role_name, vehicle_index):
    selectors_given = sum(v is not None for v in (vehicle_id, vehicle_role_name, vehicle_index))
    if selectors_given == 0:
        raise ValueError(
            "SensorRecorder requires exactly one of vehicle_index, vehicle_id, "
            "or vehicle_role_name. Available vehicles: {}".format(list_vehicles(world))
        )
    if selectors_given > 1:
        raise ValueError(
            "SensorRecorder: specify only ONE of vehicle_index, vehicle_id, "
            "vehicle_role_name -- got more than one."
        )

    candidates = _sorted_vehicle_actors(world)

    if vehicle_index is not None:
        if vehicle_index < 0 or vehicle_index >= len(candidates):
            raise ValueError(
                "No vehicle with index={}: there are {} vehicles in the scene, "
                "valid indices are 0..{}. Available vehicles: {}".format(
                    vehicle_index, len(candidates), len(candidates) - 1, list_vehicles(world))
            )
        return candidates[vehicle_index]

    if vehicle_id is not None:
        for actor in candidates:
            if actor.id == vehicle_id:
                return actor
        raise ValueError(
            "No vehicle with id={} found. Available vehicles: {}".format(
                vehicle_id, list_vehicles(world))
        )

    matches = [a for a in candidates if a.attributes.get("role_name", "") == vehicle_role_name]
    if not matches:
        raise ValueError(
            "No vehicle with role_name='{}' found. Available vehicles: {}".format(
                vehicle_role_name, list_vehicles(world))
        )
    if len(matches) > 1:
        raise ValueError(
            "role_name='{}' is ambiguous ({} matches). Use vehicle_index or vehicle_id instead. "
            "Available vehicles: {}".format(vehicle_role_name, len(matches), list_vehicles(world))
        )
    return matches[0]


def _camera_intrinsic(image_size_x, image_size_y, fov_degrees):
    # Matches the DeepAccident dataset's intrinsic layout exactly (not the
    # standard OpenCV [[fx,0,cx],[0,fy,cy],[0,0,1]] convention):
    #   [[cx, fx,  0], [cy, 0, -fx], [1, 0, 0]]
    focal = image_size_x / (2.0 * math.tan(fov_degrees * math.pi / 360.0))
    cx = image_size_x / 2.0
    cy = image_size_y / 2.0
    K = np.array([
        [cx, focal, 0.0],
        [cy, 0.0, -focal],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)
    return K


def _transform_matrix(transform):
    return np.array(transform.get_matrix(), dtype=np.float64)


def _inverse_transform_matrix(transform):
    return np.array(transform.get_inverse_matrix(), dtype=np.float64)


class SensorRecorder(object):
    """
    Attaches the chosen sensor rig(s) -- the 6-camera surround set, the
    single LiDAR, or both together -- to ONE chosen vehicle, and records
    output + calibration to disk, frame-synchronized to the CARLA
    simulation tick.
    """

    def __init__(self, world, output_dir, scenario_name, fps, sensor_type,
                 vehicle_index=None, vehicle_id=None, vehicle_role_name=None):
        if sensor_type not in SENSOR_TYPES:
            raise ValueError("sensor_type must be one of {}, got '{}'".format(
                SENSOR_TYPES, sensor_type))

        self.world = world
        self.fps = fps
        self.sensor_type = sensor_type
        self.want_cameras = sensor_type in ("cameras", "both")
        self.want_lidar = sensor_type in ("lidar", "both")

        self.vehicle = _resolve_vehicle(world, vehicle_id, vehicle_role_name, vehicle_index)
        role_name = self.vehicle.attributes.get("role_name", "")
        self.vehicle_label = role_name if role_name else "vehicle_{}".format(self.vehicle.id)

        # <output_dir>/<scenario_name>/<vehicle_label>/{Camera_*, lidar01, calib}
        self.output_dir = os.path.join(output_dir, scenario_name, self.vehicle_label)
        self.calib_dir = os.path.join(self.output_dir, "calib")

        self._sensors = []          # spawned carla.Actor sensors
        self._camera_queues = {}    # name -> FIFO list of (frame, np.ndarray(H,W,3) RGB)
        self._lidar_queue = []      # FIFO list of (frame, np.ndarray(N,4))
        self._static_calib = {}     # intrinsics / extrinsics, computed once
        self._frame_idx = 0         # written-frame counter (6-digit index)

        # Diagnostics -- printed by destroy() so a silent "nothing recorded"
        # run is easy to root-cause instead of guessing.
        self._tick_calls = 0
        self._camera_frames_received = {}  # name -> count
        self._lidar_frames_received = 0
        self._write_errors = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_sensors(self):
        blueprint_library = self.world.get_blueprint_library()
        os.makedirs(self.calib_dir, exist_ok=True)

        # Full combined calib (all 6 intrinsics + all 6 lidar_to_camera +
        # lidar_to_ego) is computed from the FIXED mount geometry constants,
        # independent of which sensor_type is actually spawned this run --
        # carla.Transform.get_matrix()/get_inverse_matrix() work on any
        # Transform, no spawned actor required. This matches the combined
        # calib.pkl format (single file per frame, not split per sensor type).
        self._static_calib = self._compute_full_static_calib()

        if self.want_cameras:
            self._setup_cameras(blueprint_library)
        if self.want_lidar:
            self._setup_lidar(blueprint_library)

    def _compute_full_static_calib(self):
        lidar_x, lidar_y, lidar_z = LIDAR_LOCATION
        lidar_transform = carla.Transform(
            carla.Location(x=lidar_x, y=lidar_y, z=lidar_z),
            carla.Rotation(yaw=LIDAR_YAW),
        )
        lidar_to_ego = _transform_matrix(lidar_transform)

        calib = {"lidar_to_ego": lidar_to_ego}

        for name, x, y, z, yaw, fov in CAMERA_CONFIGS:
            x, y, z = _camera_mount_offset(self.vehicle, x, y, z)
            camera_transform = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(yaw=yaw),
            )
            calib["intrinsic_{}".format(name)] = _camera_intrinsic(IMAGE_SIZE_X, IMAGE_SIZE_Y, fov)
            # lidar-local point -> ego frame (lidar_transform) -> camera-local frame (inverse of camera_transform)
            calib["lidar_to_{}".format(name)] = _inverse_transform_matrix(camera_transform).dot(lidar_to_ego)

        return calib

    def _setup_cameras(self, blueprint_library):
        for name, x, y, z, yaw, fov in CAMERA_CONFIGS:
            os.makedirs(os.path.join(self.output_dir, name), exist_ok=True)

            cam_bp = blueprint_library.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(IMAGE_SIZE_X))
            cam_bp.set_attribute("image_size_y", str(IMAGE_SIZE_Y))
            cam_bp.set_attribute("fov", str(fov))
            cam_bp.set_attribute("sensor_tick", str(1.0 / self.fps))

            mount_x, mount_y, mount_z = _camera_mount_offset(self.vehicle, x, y, z)
            relative_transform = carla.Transform(
                carla.Location(x=mount_x, y=mount_y, z=mount_z),
                carla.Rotation(yaw=yaw),
            )
            sensor = self.world.spawn_actor(cam_bp, relative_transform, attach_to=self.vehicle)
            self._sensors.append(sensor)
            self._camera_queues[name] = []
            self._camera_frames_received[name] = 0

            sensor.listen(self._make_camera_callback(name))

    def _make_camera_callback(self, name):
        def callback(image):
            buf = np.frombuffer(image.raw_data, dtype=np.uint8)
            buf = buf.reshape((image.height, image.width, 4))
            rgb = buf[:, :, :3][:, :, ::-1]  # BGRA -> RGB
            self._camera_queues[name].append((image.frame, rgb.copy()))
            self._camera_frames_received[name] += 1
        return callback

    def _setup_lidar(self, blueprint_library):
        os.makedirs(os.path.join(self.output_dir, "lidar01"), exist_ok=True)

        # A CARLA LiDAR only completes a full 360-degree rotation per saved
        # frame if rotation_frequency == the WORLD's actual tick rate (i.e.
        # one full revolution happens during each world.tick()). Deriving it
        # from world.get_settings().fixed_delta_seconds -- instead of trusting
        # the recorder's own `fps` argument to already match it -- means the
        # two can never silently drift apart (that drift is exactly what
        # previously caused each saved frame to only cover a partial sweep:
        # rotation_frequency ended up a fraction of the real tick rate).
        settings = self.world.get_settings()
        if not settings.synchronous_mode or not settings.fixed_delta_seconds:
            raise RuntimeError(
                "SensorRecorder requires the CARLA world to be running in "
                "synchronous mode with a fixed fixed_delta_seconds. Without "
                "it, the LiDAR's rotation speed drifts relative to the "
                "simulation tick and every saved frame only captures a "
                "partial sweep instead of a full 360-degree rotation."
            )
        world_tick_hz = 1.0 / settings.fixed_delta_seconds
        if abs(world_tick_hz - float(self.fps)) > 1e-3:
            print(
                "[SensorRecorder] warning: fps={} does not match the world's "
                "actual tick rate {:.3f} Hz (fixed_delta_seconds={}); using "
                "the world's real tick rate for rotation_frequency so each "
                "saved LiDAR frame is still a full 360-degree sweep.".format(
                    self.fps, world_tick_hz, settings.fixed_delta_seconds))
        rotation_frequency = world_tick_hz

        points_per_second = LIDAR_POINTS_PER_ROTATION * rotation_frequency

        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(LIDAR_CHANNELS))
        lidar_bp.set_attribute("range", str(LIDAR_RANGE))
        lidar_bp.set_attribute("points_per_second", str(int(points_per_second)))
        lidar_bp.set_attribute("rotation_frequency", str(rotation_frequency))
        lidar_bp.set_attribute("upper_fov", str(LIDAR_UPPER_FOV))
        lidar_bp.set_attribute("lower_fov", str(LIDAR_LOWER_FOV))
        # No sensor_tick override here: leaving it at the CARLA default fires
        # the callback on every world tick, which is exactly one full
        # rotation per callback at the rotation_frequency set above. Setting
        # sensor_tick separately is redundant when it should equal the tick
        # period anyway, and only adds another way for the two to drift apart.

        x, y, z = LIDAR_LOCATION
        relative_transform = carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(yaw=LIDAR_YAW),
        )
        sensor = self.world.spawn_actor(lidar_bp, relative_transform, attach_to=self.vehicle)
        self._sensors.append(sensor)

        sensor.listen(self._lidar_callback)

    def _lidar_callback(self, lidar_data):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        points = points.reshape((-1, 4)).astype(np.float64)  # x, y, z, intensity

        if self._lidar_frames_received == 0:
            self._check_lidar_sweep_coverage(points)

        self._lidar_queue.append((lidar_data.frame, points.copy()))
        self._lidar_frames_received += 1

    @staticmethod
    def _check_lidar_sweep_coverage(points):
        if len(points) < 50:
            return
        azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0])) % 360.0
        span = azimuth.max() - azimuth.min()
        if span < 300.0:
            print(
                "[SensorRecorder] WARNING: first LiDAR frame only spans ~{:.0f} "
                "degrees of azimuth -- expected a full ~360 degree sweep per "
                "saved frame. The world may not be in synchronous mode with "
                "fixed_delta_seconds matching this recorder's fps.".format(span))

    # ------------------------------------------------------------------
    # Per-tick
    # ------------------------------------------------------------------

    def on_tick(self):
        """
        Call once per world.tick() from the existing scenario tick loop.
        Writes out the next complete, synchronized sample (if any).
        """
        self._tick_calls += 1
        try:
            self._try_write_sample()
        except Exception as e:
            self._write_errors += 1
            print("[SensorRecorder] write error at index {}: {}".format(self._frame_idx, e))

    def _try_write_sample(self):
        # Pair modalities by ARRIVAL ORDER (FIFO), not exact frame-number
        # equality. CARLA's per-sensor sensor_tick scheduling can drift a
        # sensor out of phase with its siblings by a tick or two over a long
        # run, even though all were spawned in the same frozen sim-time
        # window -- an exact-frame match would then never recur. Treating
        # "the Nth sample from each active sensor" as one synchronized frame
        # is robust to that. Only write once every WANTED modality has data.
        if self.want_cameras and not all(self._camera_queues[name] for name, *_ in CAMERA_CONFIGS):
            return
        if self.want_lidar and not self._lidar_queue:
            return

        ego_to_world = _transform_matrix(self.vehicle.get_transform())
        frames_used = []

        if self.want_cameras:
            for name, *_ in CAMERA_CONFIGS:
                cam_frame, img = self._camera_queues[name].pop(0)
                frames_used.append(cam_frame)
                path = os.path.join(self.output_dir, name, "{:06d}.jpg".format(self._frame_idx))
                Image.fromarray(img, mode="RGB").save(path, quality=95)

        if self.want_lidar:
            # Since the LiDAR now fires every world tick (see _setup_lidar) to
            # guarantee a clean, undistorted full rotation per callback, it
            # fills up faster than a slower fps's camera-gated write cadence
            # can drain it. Always take the FRESHEST queued sweep and discard
            # the rest -- otherwise every write would pop the oldest
            # backlogged item, and that backlog (and its staleness relative
            # to the camera/pose captured this tick) would only grow over
            # the course of a recording.
            while len(self._lidar_queue) > 1:
                self._lidar_queue.pop(0)
            lidar_frame, points = self._lidar_queue.pop(0)
            frames_used.append(lidar_frame)
            path = os.path.join(self.output_dir, "lidar01", "{:06d}.npz".format(self._frame_idx))
            np.savez(path, data=points)

        spread = max(frames_used) - min(frames_used)
        if spread > 2:
            print("[SensorRecorder] warning: frame spread={} for index {} (frames={})".format(
                spread, self._frame_idx, frames_used))

        self._write_calib(ego_to_world)
        self._frame_idx += 1

    def _write_calib(self, ego_to_world):
        calib = dict(self._static_calib)
        calib["ego_to_world"] = ego_to_world
        path = os.path.join(self.calib_dir, "{:06d}.pkl".format(self._frame_idx))
        with open(path, "wb") as f:
            pickle.dump(calib, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def destroy(self):
        received = dict(self._camera_frames_received)
        if self.want_lidar:
            received["lidar01"] = self._lidar_frames_received
        print("[SensorRecorder] summary: on_tick calls={}, frames written={}, "
              "write errors={}, sensor callbacks received={}".format(
                  self._tick_calls, self._frame_idx, self._write_errors, received))

        for sensor in self._sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._sensors = []
        self._camera_queues = {}
        self._lidar_queue = []
