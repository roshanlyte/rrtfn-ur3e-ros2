import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import random
import math

JOINT_LIMITS = [(-2*math.pi, 2*math.pi)] * 6
JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

class Node_:
    def __init__(self, config):
        self.config = np.array(config)
        self.parent = None
        self.cost = 0.0
        self.children = []

def distance(a, b):
    return np.linalg.norm(a.config - b.config)

def sample_random(goal_config, goal_bias=0.1):
    if random.random() < goal_bias:
        return Node_(goal_config)
    return Node_([random.uniform(lo, hi) for lo, hi in JOINT_LIMITS])

def nearest(tree, sample):
    return min(tree, key=lambda n: distance(n, sample))

def steer(from_node, to_node, step_size=0.1):
    direction = to_node.config - from_node.config
    dist = np.linalg.norm(direction)
    if dist < step_size:
        return Node_(to_node.config.copy())
    return Node_(from_node.config + (direction / dist) * step_size)

def is_collision_free(from_node, to_node):
    return True

def near_nodes(tree, new_node, radius=0.5):
    return [n for n in tree if distance(n, new_node) <= radius]

def choose_parent(neighbours, nearest_node, new_node):
    best_parent = nearest_node
    best_cost = nearest_node.cost + distance(nearest_node, new_node)
    for node in neighbours:
        if is_collision_free(node, new_node):
            cost = node.cost + distance(node, new_node)
            if cost < best_cost:
                best_cost = cost
                best_parent = node
    new_node.cost = best_cost
    return best_parent

def rewire(tree, neighbours, new_node):
    for node in neighbours:
        new_cost = new_node.cost + distance(new_node, node)
        if new_cost < node.cost and is_collision_free(new_node, node):
            node.parent.children.remove(node)
            node.parent = new_node
            node.cost = new_cost
            new_node.children.append(node)

def find_worst_leaf(tree):
    leaves = [n for n in tree if len(n.children) == 0]
    return max(leaves, key=lambda n: n.cost)

def extract_path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append(node.config.tolist())
        node = node.parent
    return list(reversed(path))

def rrtfn(start_config, goal_config, max_nodes=500, max_iter=2000,
          step_size=0.1, goal_radius=0.15, goal_bias=0.1):
    start = Node_(start_config)
    goal = Node_(goal_config)
    tree = [start]
    goal_node = None
    for i in range(max_iter):
        sample = sample_random(goal_config, goal_bias)
        near = nearest(tree, sample)
        new_node = steer(near, sample, step_size)
        if not is_collision_free(near, new_node):
            continue
        neighbours = near_nodes(tree, new_node, radius=0.5)
        best_parent = choose_parent(neighbours, near, new_node)
        new_node.parent = best_parent
        best_parent.children.append(new_node)
        tree.append(new_node)
        rewire(tree, neighbours, new_node)
        if len(tree) > max_nodes:
            worst = find_worst_leaf(tree)
            if worst.parent:
                worst.parent.children.remove(worst)
            tree.remove(worst)
        if distance(new_node, goal) < goal_radius:
            if goal_node is None or new_node.cost < goal_node.cost:
                goal_node = new_node
    if goal_node:
        return extract_path(goal_node)
    return None

def interpolate_path(path, points_per_segment=50):
    smooth = []
    for i in range(len(path) - 1):
        a = np.array(path[i])
        b = np.array(path[i+1])
        for t in np.linspace(0, 1, points_per_segment):
            smooth.append((a + t * (b - a)).tolist())
    return smooth

class RRTFNPlannerNode(Node):
    def __init__(self):
        super().__init__('rrtfn_planner')
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)
        self.current_config = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
        self.trajectory = []
        self.traj_index = 0
        self.get_logger().info('RRT*FN Planner Node started!')
        self.publish_timer = self.create_timer(0.02, self.publish_joint_state)
        self.plan_timer = self.create_timer(2.0, self.plan_once)

    def publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        if self.trajectory and self.traj_index < len(self.trajectory):
            self.current_config = self.trajectory[self.traj_index]
            self.traj_index += 1
        msg.position = self.current_config
        self.publisher_.publish(msg)

    def plan_once(self):
        self.plan_timer.cancel()
        start = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
        goal  = [1.0, -1.0, 0.5, -1.0, 0.5, 0.3]
        self.get_logger().info('Running RRT*FN...')
        path = rrtfn(start, goal, max_nodes=500, max_iter=3000)
        if path is None:
            self.get_logger().error('No path found!')
            return
        self.get_logger().info(f'Path found! {len(path)} waypoints.')
        self.trajectory = interpolate_path(path, points_per_segment=50)
        self.traj_index = 0
        self.get_logger().info(f'Animating {len(self.trajectory)} frames...')

def main(args=None):
    rclpy.init(args=args)
    node = RRTFNPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
