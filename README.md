# RRT*FN Path Planner for UR3e (ROS2 Jazzy)

Final year project implementing the **RRT*FN (Rapidly-exploring Random Tree Star Fixed Nodes)** path planning algorithm for the Universal Robots UR3e collaborative manipulator, with benchmarking against RRT and RRT*.

**Author:** Roshan Ashraf
**Supervisor:** Sotirios
**Institution:** University of Portsmouth — BEng Final Year Project

---

## What's in this repository

| File | Description |
|------|-------------|
| `rrtfn_planner/algorithms.py` | Implementations of RRT, RRT*, and RRT*FN sharing a common collision checker |
| `rrtfn_planner/rrtfn_node.py` | Live ROS2 node: plans paths in response to incoming goal positions, publishes joint states for RViz |
| `rrtfn_planner/arm_tracker.py` | MediaPipe-based webcam node that publishes hand position as a goal target |
| `rrtfn_planner/collision.py` | Sphere-based collision checker using ikpy forward kinematics |
| `rrtfn_planner/benchmark.py` | Runs all three algorithms across multiple scenarios and logs results to CSV |

---

## System architecture

