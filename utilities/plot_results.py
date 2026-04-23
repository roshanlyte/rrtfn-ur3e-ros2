import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path.home() / 'rrtfn_benchmark_with_obstacles.csv'
OUTPUT_DIR = Path.home() / 'rrtfn_plots'
OUTPUT_DIR.mkdir(exist_ok=True)

rows = []
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['success']     = row['success'].lower() == 'true'
        row['path_length'] = float(row['path_length'])
        row['tree_size']   = int(row['tree_size'])
        row['time_s']      = float(row['time_s'])
        rows.append(row)

def group_metric(metric, only_successful=True):
    data = defaultdict(list)
    for r in rows:
        if only_successful and not r['success']:
            continue
        if metric == 'path_length' and r['path_length'] == float('inf'):
            continue
        data[(r['scenario'], r['planner'])].append(r[metric])
    return data

scenarios = ['overhead_reach','wide_sweep','reach_around_box','high_to_low']
planners  = ['RRT', 'RRT*', 'RRT*FN']
colors    = {'RRT': '#94a3b8', 'RRT*': '#0ea5e9', 'RRT*FN': '#10b981'}

def grouped_bar_chart(metric_name, ylabel, title, logscale=False, filename='plot.png'):
    data = group_metric(metric_name)
    means = np.zeros((len(planners), len(scenarios)))
    stds  = np.zeros((len(planners), len(scenarios)))
    for i, p in enumerate(planners):
        for j, s in enumerate(scenarios):
            vals = data.get((s, p), [])
            if vals:
                means[i, j] = np.mean(vals)
                stds[i, j]  = np.std(vals)
    x = np.arange(len(scenarios))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, p in enumerate(planners):
        ax.bar(x + (i - 1) * w, means[i], w, yerr=stds[i],
               label=p, color=colors[p], capsize=4, edgecolor='white')
    ax.set_xlabel('Scenario')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15)
    ax.legend(title='Planner')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    if logscale:
        ax.set_yscale('log')
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved {path}')

grouped_bar_chart('path_length', 'Path length (radians)',
    'Planned path length by scenario (lower is better)', filename='01_path_length.png')
grouped_bar_chart('time_s', 'Computation time (s)',
    'Computation time by scenario (log scale, lower is better)', logscale=True, filename='02_computation_time.png')
grouped_bar_chart('tree_size', 'Tree size (nodes)',
    'Final tree size by scenario (lower = less memory)', filename='03_tree_size.png')

fig, ax = plt.subplots(figsize=(9, 5))
for p in planners:
    xs = [r['tree_size'] for r in rows if r['planner'] == p and r['success']]
    ys = [r['path_length'] for r in rows if r['planner'] == p and r['success']]
    ax.scatter(xs, ys, label=p, color=colors[p], s=60, alpha=0.7, edgecolor='white')
ax.set_xlabel('Tree size (nodes)')
ax.set_ylabel('Path length (radians)')
ax.set_title('Quality vs memory tradeoff (bottom-left is best)')
ax.legend(title='Planner')
ax.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_quality_vs_memory.png', dpi=150)
plt.close()
print(f'Saved {OUTPUT_DIR}/04_quality_vs_memory.png')
print(f'\nAll plots saved to: {OUTPUT_DIR}')
