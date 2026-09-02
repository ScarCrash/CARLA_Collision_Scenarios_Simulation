# CARLA Collision Sensor Recorder

Two hand-built [CARLA](https://carla.org/) + [scenario_runner](https://github.com/carla-simulator/scenario_runner)
traffic-collision scenarios, each wired up to record synchronized multi-camera + LiDAR sensor
data (with full calibration) from **any single vehicle you choose** — the ego, either colliding
vehicle, or any of ~30 background traffic vehicles.

Output contains 6 surround cameras, 1 top LiDAR, and a combined per-frame calibration file.

## Recorded Data

Pre-recorded sensor data for both scenarios is available on the
[Releases page](https://github.com/siribooncha/CARLA_Collision_Scenarios_Simulation/releases/tag/release-assets):

- `five-way-signalised-intersection-collision-data.zip`
- `four-way-unsignalised-intersection-collision-data-part1.zip` + `four-way-unsignalised-intersection-collision-data-part2.zip`
  (split across 2 files due to GitHub's per-file size limit — extract both into the same destination folder)

Download and extract into a `sensor_output/` folder at the repo root to match the layout described
below.

## Scenarios

| Scenario | Map | Description |
|---|---|---|
| **5-way signalised intersection collision** | Town03 | 2 vehicles collide at a 5-way signalised intersection, ~30 background vehicles nearby |
| **4-way unsignalised intersection collision** | Town04 | 3 vehicles collide near a 4-way unsignalised intersection, ~30 background vehicles nearby |


https://github.com/user-attachments/assets/81b95d41-0c02-428a-8e00-77794f1fa9f1

https://github.com/user-attachments/assets/97087b87-ce0c-424f-b7a5-f2ad7f0599ec

Both scenarios run an ego vehicle that drives through and observes the crash. Every vehicle in
the scene — ego, colliding, or background — can be picked as the recording target at launch time,
no code edits required.

Each scenario ends automatically **2 simulated seconds after the first collision is detected**
(or, if no collision occurs, once the ego finishes its drive distance as a fallback) — so a run
captures the lead-up and immediate aftermath of the crash without recording indefinitely.

## Requirements

- CARLA simulator (tested on 0.9.14) 
- scenario_runner (tested on 0.9.13)
- Python 3.7 

## Setup

Drop these files into your `scenario_runner` checkout — the recorder module needs to be
importable as `sensor_recorder`, and the two scenario scripts need to be discoverable by
scenario_runner's `--additionalScenario` flag:

```
<scenario_runner root>/
    carla-collision-sensor-recorder/
        sensor_recorder.py
        five_way_signalised_intersection_collision.py
        five_way_signalised_intersection_collision.xml
        four_way_unsignalised_intersection_collision.py
        four_way_unsignalised_intersection_collision.xml
```

No changes to scenario_runner itself are needed — everything here is self-contained.

## Running a scenario

Start the CARLA server first, then from the scenario_runner root:

```bat
python scenario_runner.py --scenario FiveWaySignalisedIntersectionCollision ^
    --configFile carla-collision-sensor-recorder/five_way_signalised_intersection_collision.xml ^
    --additionalScenario carla-collision-sensor-recorder/five_way_signalised_intersection_collision.py ^
    --sync --reloadWorld --timeout 60
```

```bat
python scenario_runner.py --scenario FourWayUnsignalisedIntersectionCollision ^
    --configFile carla-collision-sensor-recorder/four_way_unsignalised_intersection_collision.xml ^
    --additionalScenario carla-collision-sensor-recorder/four_way_unsignalised_intersection_collision.py ^
    --sync --reloadWorld --timeout 60
```

`--sync` and `--reloadWorld` are required — the recorder relies on a synchronous, fixed-tick-rate
world, and a fresh map load avoids leftover actors from a previous run blocking spawn points.

On Linux/macOS, replace the trailing `^` line continuations with `\`.

## Choosing what to record

Set these environment variables before launching (all optional, sensible defaults shown):

| Variable | Default | Meaning |
|---|---|---|
| `RECORD_VEHICLE_INDEX` | `0` (ego) | 0-based index into the vehicle list printed at scenario start |
| `RECORD_VEHICLE_ROLE_NAME` | — | Select by role name instead of index, e.g. `lidar_target`, `camera_target` |
| `RECORD_SENSOR_TYPE` | `cameras` | `cameras`, `lidar`, or `both` |
| `RECORD_OUTPUT_DIR` | `sensor_output` | Output root directory |
| `RECORD_FPS` | `20.0` | Capture rate in Hz — should not exceed the world's tick rate |

Example — record the LiDAR-tagged background vehicle with both cameras and LiDAR at 5 Hz:

```bat
set RECORD_VEHICLE_ROLE_NAME=lidar_target
set RECORD_SENSOR_TYPE=both
set RECORD_FPS=5
python scenario_runner.py --scenario FiveWaySignalisedIntersectionCollision ^
    --configFile carla-collision-sensor-recorder/five_way_signalised_intersection_collision.xml ^
    --additionalScenario carla-collision-sensor-recorder/five_way_signalised_intersection_collision.py ^
    --sync --reloadWorld --timeout 60
```

Every run prints the full vehicle list at startup:

```
[SensorRecorder] Vehicles available for recording (0..33):
    index=0   id=227  role_name='hero'      type=vehicle.tesla.model3
    index=1   id=228  role_name='autopilot' type=vehicle.tesla.model3
    ...
```

Use this to confirm your `RECORD_VEHICLE_INDEX`/`RECORD_VEHICLE_ROLE_NAME` resolves to the
vehicle you actually want — the exact count/order can shift slightly run to run if a spawn
point happens to be blocked.

**Vehicle index layout** (typical; always confirm against the printed list):

- **5-way signalised intersection collision**: `N=0` ego, `N=1,2` the 2 colliding vehicles, `N=3..30` background traffic (30 total)
- **4-way unsignalised intersection collision**: `N=0` ego, `N=1,2,3` the 3 colliding vehicles (red sedan, green mini, blue truck), `N=4..33` background traffic (30 total)

## Output format

```
<RECORD_OUTPUT_DIR>/<ScenarioName>/<vehicle_label>/
    Camera_Front/000000.jpg, 000001.jpg, ...
    Camera_FrontLeft/...
    Camera_FrontRight/...
    Camera_Back/...
    Camera_BackLeft/...
    Camera_BackRight/...
    lidar01/000000.npz, 000001.npz, ...
    calib/000000.pkl, 000001.pkl, ...
```

`<vehicle_label>` is the recorded vehicle's `role_name` if it has one (e.g. `hero`,
`lidar_target`, `camera_target`), otherwise `vehicle_<id>`.

- **Camera images**: plain RGB JPEG, 1600x900.
- **LiDAR** (`lidar01/*.npz`): single key `"data"`, a `float64` array of shape `(N, 4)` —
  columns `[x, y, z, intensity]` in the LiDAR's local frame.
- **Calibration** (`calib/*.pkl`): one combined file per frame (Python pickle, dict of numpy
  arrays), matching the DeepAccident format:

  | Key | Shape | Notes |
  |---|---|---|
  | `ego_to_world` | `(4,4)` | Per-frame, the recorded vehicle's pose in world coordinates |
  | `intrinsic_Camera_*` (x6) | `(3,3)` | Static; DeepAccident's intrinsic layout `[[cx,fx,0],[cy,0,-fx],[1,0,0]]`, not the standard OpenCV convention |
  | `lidar_to_Camera_*` (x6) | `(4,4)` | Static; LiDAR-frame -> camera-frame transform |
  | `lidar_to_ego` | `(4,4)` | Static; LiDAR mount transform relative to the vehicle |

  All 13 keys are always present regardless of `RECORD_SENSOR_TYPE`, since they're derived from
  fixed mount geometry, not from which sensors are actually spawned that run.

Frame indices are a 6-digit, 0-based, zero-padded counter, identical across every modality for a
given tick (`Camera_Front/000005.jpg`, `lidar01/000005.npz`, and `calib/000005.pkl` are all the
same simulated instant).

## Tuning

- **LiDAR density**: `LIDAR_POINTS_PER_ROTATION` in `sensor_recorder.py` (default `37500` rays
  cast per rotation — the actual point count in each `.npz` will usually be lower, since only
  rays that hit something return a point).
- **Camera/LiDAR mount positions**: `CAMERA_CONFIGS` / `LIDAR_LOCATION` in `sensor_recorder.py`.
- **Background vehicle count**: `TARGET_SURROUND_VEHICLES` near the top of each scenario's
  `_initialize_actors()`.

If you push `RECORD_SENSOR_TYPE=both` with a high `RECORD_FPS` and/or a very high
`LIDAR_POINTS_PER_ROTATION`, you may hit a simulator timeout (`RuntimeError: time-out ... while
waiting for the simulator`) — that's real per-tick render/ray-cast load, not a bug. Lower
`RECORD_FPS`, lower the LiDAR density, or raise `--timeout` on the command line.

## How it works (brief)

- Both scenarios override `_setup_scenario_trigger()` to skip scenario_runner's default
  route-arrival trigger, which would otherwise deadlock (it needs the ego to already be moving
  to succeed, but the ego only starts moving *after* the trigger succeeds).
- All actors spawn synchronously in `_initialize_actors()` (not via delayed/threaded spawning),
  and cleanup runs from `remove_all_actors()` — the actual hook `scenario_runner.py` calls when a
  run finishes — including a self-heal step that destroys any leftover vehicles from a
  previous interrupted run before spawning new ones.
- `SensorRecorder` (in `sensor_recorder.py`) attaches the requested sensor rig to the chosen
  vehicle and hooks into the scenario's existing tick loop via a small `py_trees` behavior — it
  does not run its own `world.tick()` loop.
- Background vehicle spawn points are drawn from a hand-picked list near the crash site, with a
  fallback to the map's own recommended spawn points (`world.get_map().get_spawn_points()`) for
  any point that fails to spawn, and a top-up pass that fills in extra background vehicles from
  nearby spawn points until the target count is reached.
