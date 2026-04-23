"""
Simple collision checker for UR3e path planning.
Robot: spheres along each link.
Obstacles: axis-aligned boxes.
"""
import numpy as np
import ikpy.chain

URDF_PATH = "/home/knight/ur3e_fixed.urdf"
UR3E_CHAIN = ikpy.chain.Chain.from_urdf_file(
    URDF_PATH, base_elements=["base_link"]
)

# (chain_link_index, sphere_radius_m)
# Indices 2-7 are the 6 joints
ROBOT_SPHERES = [
    (2, 0.08),
    (3, 0.07),
    (4, 0.06),
    (5, 0.05),
    (6, 0.05),
    (7, 0.05),
]


class Obstacle:
    def __init__(self, name, center, half_sizes):
        self.name = name
        self.center = np.array(center, dtype=float)
        self.half_sizes = np.array(half_sizes, dtype=float)

    def min_distance_to_point(self, point):
        p = np.array(point, dtype=float)
        d = np.maximum(np.abs(p - self.center) - self.half_sizes, 0.0)
        return np.linalg.norm(d)


def joints_to_link_positions(joint_config):
    """
    Return 3D (x,y,z) position of each tracked link.
    Uses ikpy forward_kinematics which returns ONE transformation matrix
    for the end of the chain — we compute per-link via successive calls.
    """
    full = [0.0, 0.0] + list(joint_config) + [0.0]
    positions = []
    # Build a truncated chain config for each link of interest
    for link_idx, _radius in ROBOT_SPHERES:
        # Mask: only include links up to this index
        config = [0.0] * len(UR3E_CHAIN.links)
        for i in range(link_idx + 1):
            config[i] = full[i]
        T = UR3E_CHAIN.forward_kinematics(config)
        positions.append(T[:3, 3])
    return positions


def config_in_collision(joint_config, obstacles):
    link_positions = joints_to_link_positions(joint_config)
    for (link_idx, radius), pos in zip(ROBOT_SPHERES, link_positions):
        for obs in obstacles:
            if obs.min_distance_to_point(pos) < radius:
                return True
    return False


def edge_in_collision(config_a, config_b, obstacles, resolution=0.1):
    """
    Check an edge between two joint configs by interpolating.
    Larger resolution = faster but less accurate.
    """
    a = np.array(config_a, dtype=float)
    b = np.array(config_b, dtype=float)
    step = np.linalg.norm(b - a)
    n_checks = max(2, int(np.ceil(step / resolution)))
    for i in range(n_checks + 1):
        t = i / n_checks
        config = a + t * (b - a)
        if config_in_collision(config, obstacles):
            return True
    return False


# Default obstacle set (metres, robot base frame)
DEFAULT_OBSTACLES = [
    Obstacle("table", center=[0.3, 0.0, -0.05], half_sizes=[0.4, 0.4, 0.02]),
    Obstacle("wall_right", center=[0.3, -0.35, 0.25], half_sizes=[0.3, 0.02, 0.3]),
    Obstacle("box_on_table", center=[0.35, 0.1, 0.1], half_sizes=[0.05, 0.05, 0.1]),
]
