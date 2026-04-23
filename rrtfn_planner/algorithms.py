"""
Three planning algorithms for benchmarking:
    - RRT, RRT*, RRT*FN
All use the same collision checker so differences reflect only the algorithm.
"""
import numpy as np
import random
import math
import time

from rrtfn_planner.collision import edge_in_collision, config_in_collision, DEFAULT_OBSTACLES

JOINT_LIMITS = [(-2*math.pi, 2*math.pi)] * 6


class TreeNode:
    def __init__(self, config):
        self.config = np.array(config)
        self.parent = None
        self.cost = 0.0
        self.children = []


def distance(a, b):
    return np.linalg.norm(a.config - b.config)

def sample_random(goal_config, obstacles, goal_bias=0.1, max_attempts=50):
    """Sample a collision-free config (or fall back to random after N tries)."""
    if random.random() < goal_bias:
        return TreeNode(goal_config)
    for _ in range(max_attempts):
        cfg = [random.uniform(lo, hi) for lo, hi in JOINT_LIMITS]
        if not config_in_collision(cfg, obstacles):
            return TreeNode(cfg)
    return TreeNode(cfg)

def nearest(tree, sample):
    return min(tree, key=lambda n: distance(n, sample))

def steer(from_node, to_node, step_size=0.2):
    direction = to_node.config - from_node.config
    dist = np.linalg.norm(direction)
    if dist < step_size:
        return TreeNode(to_node.config.copy())
    return TreeNode(from_node.config + (direction / dist) * step_size)

def is_edge_free(from_node, to_node, obstacles):
    return not edge_in_collision(from_node.config, to_node.config, obstacles)

def near_nodes(tree, new_node, radius=0.5):
    return [n for n in tree if distance(n, new_node) <= radius]

def extract_path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append(node.config.tolist())
        node = node.parent
    return list(reversed(path))

def path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    return sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
               for i in range(len(path) - 1))


# ---------- RRT ----------

def rrt(start, goal, obstacles, max_iter=3000, step_size=0.2,
        goal_radius=0.25, goal_bias=0.1):
    s = TreeNode(start)
    g = TreeNode(goal)
    tree = [s]
    for i in range(max_iter):
        sample = sample_random(goal, obstacles, goal_bias)
        near = nearest(tree, sample)
        new_node = steer(near, sample, step_size)
        if not is_edge_free(near, new_node, obstacles):
            continue
        new_node.parent = near
        near.children.append(new_node)
        tree.append(new_node)
        if distance(new_node, g) < goal_radius:
            return extract_path(new_node), len(tree), i + 1
    return None, len(tree), max_iter


# ---------- RRT* ----------

def rrt_star(start, goal, obstacles, max_iter=3000, step_size=0.2,
             goal_radius=0.25, goal_bias=0.1, neighbour_radius=0.5):
    s = TreeNode(start)
    g = TreeNode(goal)
    tree = [s]
    goal_node = None
    for i in range(max_iter):
        sample = sample_random(goal, obstacles, goal_bias)
        near = nearest(tree, sample)
        new_node = steer(near, sample, step_size)
        if not is_edge_free(near, new_node, obstacles):
            continue
        neighbours = near_nodes(tree, new_node, neighbour_radius)
        best_parent = near
        best_cost = near.cost + distance(near, new_node)
        for node in neighbours:
            if is_edge_free(node, new_node, obstacles):
                c = node.cost + distance(node, new_node)
                if c < best_cost:
                    best_cost = c
                    best_parent = node
        new_node.cost = best_cost
        new_node.parent = best_parent
        best_parent.children.append(new_node)
        tree.append(new_node)
        for node in neighbours:
            new_c = new_node.cost + distance(new_node, node)
            if new_c < node.cost and is_edge_free(new_node, node, obstacles):
                if node.parent:
                    node.parent.children.remove(node)
                node.parent = new_node
                node.cost = new_c
                new_node.children.append(node)
        if distance(new_node, g) < goal_radius:
            if goal_node is None or new_node.cost < goal_node.cost:
                goal_node = new_node
    if goal_node:
        return extract_path(goal_node), len(tree), max_iter
    return None, len(tree), max_iter


# ---------- RRT*FN ----------

def find_worst_leaf(tree):
    leaves = [n for n in tree if len(n.children) == 0]
    return max(leaves, key=lambda n: n.cost)

def rrt_star_fn(start, goal, obstacles, max_iter=3000, step_size=0.2,
                goal_radius=0.25, goal_bias=0.1, neighbour_radius=0.5,
                max_nodes=300):
    s = TreeNode(start)
    g = TreeNode(goal)
    tree = [s]
    goal_node = None
    for i in range(max_iter):
        sample = sample_random(goal, obstacles, goal_bias)
        near = nearest(tree, sample)
        new_node = steer(near, sample, step_size)
        if not is_edge_free(near, new_node, obstacles):
            continue
        neighbours = near_nodes(tree, new_node, neighbour_radius)
        best_parent = near
        best_cost = near.cost + distance(near, new_node)
        for node in neighbours:
            if is_edge_free(node, new_node, obstacles):
                c = node.cost + distance(node, new_node)
                if c < best_cost:
                    best_cost = c
                    best_parent = node
        new_node.cost = best_cost
        new_node.parent = best_parent
        best_parent.children.append(new_node)
        tree.append(new_node)
        for node in neighbours:
            new_c = new_node.cost + distance(new_node, node)
            if new_c < node.cost and is_edge_free(new_node, node, obstacles):
                if node.parent:
                    node.parent.children.remove(node)
                node.parent = new_node
                node.cost = new_c
                new_node.children.append(node)
        if len(tree) > max_nodes:
            worst = find_worst_leaf(tree)
            if worst.parent:
                worst.parent.children.remove(worst)
            tree.remove(worst)
        if distance(new_node, g) < goal_radius:
            if goal_node is None or new_node.cost < goal_node.cost:
                goal_node = new_node
    if goal_node:
        return extract_path(goal_node), len(tree), max_iter
    return None, len(tree), max_iter


def run_planner(name, start, goal, seed=None, obstacles=None, **kwargs):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if obstacles is None:
        obstacles = DEFAULT_OBSTACLES

    t0 = time.perf_counter()
    if name == "RRT":
        path, tree_size, iters = rrt(start, goal, obstacles, **kwargs)
    elif name == "RRT*":
        path, tree_size, iters = rrt_star(start, goal, obstacles, **kwargs)
    elif name == "RRT*FN":
        path, tree_size, iters = rrt_star_fn(start, goal, obstacles, **kwargs)
    else:
        raise ValueError(f"Unknown planner: {name}")
    elapsed = time.perf_counter() - t0

    return {
        "planner": name,
        "success": path is not None,
        "path_length": path_length(path) if path else float("inf"),
        "num_waypoints": len(path) if path else 0,
        "tree_size": tree_size,
        "iterations": iters,
        "time_s": elapsed,
    }
