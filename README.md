# RRT*FN Path Planner for UR3e (ROS2 Jazzy)

Final year project implementing the RRT*FN path planning algorithm for the UR3e, with benchmarking against RRT and RRT*.

Author: Roshan Ashraf
Supervisor: Sotirios
University of Portsmouth — BEng Final Year Project

## Code

- rrtfn_planner/algorithms.py — RRT, RRT*, RRT*FN implementations
- rrtfn_planner/rrtfn_node.py — live planner node
- rrtfn_planner/arm_tracker.py — webcam-based arm tracker
- rrtfn_planner/collision.py — collision checking
- rrtfn_planner/benchmark.py — benchmark harness

## Other folders

- results/ — benchmark CSVs
- plots/ — figures
- utilities/ — URDF, launch script, plot generator

## Dependencies

- ROS2 Jazzy on Ubuntu 24.04
- Python: numpy<2, ikpy, mediapipe==0.10.14, opencv-python==4.10.0.84

## Key results

RRT*FN matches RRT* path quality using ~10x less memory (300 vs 3000 nodes). See plots/ and results/ for details.

## Progress

- [x] ROS2 + UR3e setup
- [x] RRT*FN from scratch
- [x] RViz trajectory execution
- [x] IK via ikpy
- [x] Webcam arm tracker
- [x] Benchmark harness with CSV
- [x] Collision checking
- [ ] Energy-aware cost function
- [ ] Jacobian-weighted sampling
- [ ] Real UR3e validation
