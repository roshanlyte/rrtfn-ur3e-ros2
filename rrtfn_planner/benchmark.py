"""
Benchmark: RRT vs RRT* vs RRT*FN with collision checking.
Scenarios chosen to avoid the start/goal-in-collision issue.
"""
import csv
from pathlib import Path
from rrtfn_planner.algorithms import run_planner
from rrtfn_planner.collision import config_in_collision, DEFAULT_OBSTACLES

SCENARIOS = [
    ("overhead_reach",
     [ 0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0],
     [ 1.0, -1.0,    0.8, -1.5,    0.0, 0.0]),
    ("wide_sweep",
     [-0.8, -1.5708, 0.0, -1.5708, 0.0, 0.0],
     [ 0.8, -1.5708, 0.0, -1.5708, 0.0, 0.0]),
    ("reach_around_box",
     [ 0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0],
     [ 0.6, -0.9,    1.0, -1.2,    0.3, 0.0]),
    ("high_to_low",
     [ 0.5, -0.5,    0.3, -1.5708, 0.0, 0.0],
     [ 0.5, -2.2,    1.5, -1.5708, 0.0, 0.0]),
]

PLANNERS = ["RRT", "RRT*", "RRT*FN"]
RUNS_PER_SCENARIO = 10
OUTPUT_CSV = Path.home() / "rrtfn_benchmark_with_obstacles.csv"


def validate_scenarios():
    bad = []
    for name, start, goal in SCENARIOS:
        if config_in_collision(start, DEFAULT_OBSTACLES):
            bad.append(f"{name}: START in collision")
        if config_in_collision(goal, DEFAULT_OBSTACLES):
            bad.append(f"{name}: GOAL in collision")
    if bad:
        print("WARNING — infeasible scenarios:")
        for b in bad:
            print(f"  {b}")
        print()


def main():
    validate_scenarios()
    rows = []
    print(f"Running {len(SCENARIOS)} scenarios x {len(PLANNERS)} planners x {RUNS_PER_SCENARIO} runs")
    print(f"Total: {len(SCENARIOS) * len(PLANNERS) * RUNS_PER_SCENARIO} invocations\n")

    for scenario_name, start, goal in SCENARIOS:
        print(f"=== {scenario_name} ===")
        for planner_name in PLANNERS:
            successes, total_len, total_time, total_tree = 0, 0.0, 0.0, 0
            for run in range(RUNS_PER_SCENARIO):
                seed = hash((scenario_name, planner_name, run)) & 0xFFFFFFFF
                result = run_planner(planner_name, start, goal, seed=seed)
                result["scenario"] = scenario_name
                result["run"] = run
                rows.append(result)
                if result["success"]:
                    successes += 1
                    total_len += result["path_length"]
                total_time += result["time_s"]
                total_tree += result["tree_size"]
            avg_len = total_len / successes if successes > 0 else float("nan")
            avg_time = total_time / RUNS_PER_SCENARIO
            avg_tree = total_tree / RUNS_PER_SCENARIO
            success_rate = 100.0 * successes / RUNS_PER_SCENARIO
            print(f"  {planner_name:<8}  success: {success_rate:5.1f}%  "
                  f"avg path: {avg_len:.3f}  avg time: {avg_time:.2f}s  "
                  f"avg tree: {avg_tree:.0f}")
        print()

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario","planner","run","success",
            "path_length","num_waypoints","tree_size","iterations","time_s"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
