import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
import numpy as np
import random
import math
import ikpy.chain

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
JOINT_LIMITS = [(-2*math.pi, 2*math.pi)] * 6

# Load the UR3e kinematic chain once at startup
URDF_PATH = '/home/knight/ur3e_fixed.urdf'
UR3E_CHAIN = ikpy.chain.Chain.from_urdf_file(
    URDF_PATH,
    base_elements=['base_link']
)

# RRT*FN core
class Node_:
    def __init__(self, config):
        self.config = np.array(config)
        self.parent = None
        self.cost = 0.0
        self.children = []

def distance(a, b):
    return np.linalg.norm(a.config - b.config)

def sample_random(goal_config, goal_bias=0.3):
    if random.random() < goal_bias:
        return Node_(goal_config)
    return Node_([random.uniform(lo, hi) for lo, hi in JOINT_LIMITS])

def nearest(tree, sample):
    return min(tree, key=lambda n: distance(n, sample))

def steer(from_node, to_node, step_size=0.2):
    direction = to_node.config - from_node.config
    dist = np.linalg.norm(direction)
    if dist < step_size:
        return Node_(to_node.config.copy())
    return Node_(from_node.config + (direction / dist) * step_size)

def is_collision_free(from_node, to_node):
    return True

def near_nodes(tree, new_node, radius=0.6):
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

def rrtfn(start_config, goal_config, max_nodes=300, max_iter=1500,
          step_size=0.2, goal_radius=0.25, goal_bias=0.3):
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
        neighbours = near_nodes(tree, new_node, radius=0.6)
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
                break
    if goal_node:
        return extract_path(goal_node)
    return None

def interpolate_path(path, points_per_segment=30):
    smooth = []
    for i in range(len(path) - 1):
        a = np.array(path[i])
        b = np.array(path[i+1])
        for t in np.linspace(0, 1, points_per_segment):
            smooth.append((a + t * (b - a)).tolist())
    return smooth

def solve_ik(target_xyz, current_joints):
    '''
    Real inverse kinematics using ikpy.
    target_xyz: [x, y, z] in robot base frame (meters)
    current_joints: list of 6 current joint angles — used as IK starting point
    Returns: list of 6 joint angles that place end-effector at target
    '''
    # ikpy chain has 9 links: [Base, base_inertia, 6 joints, ft_frame]
    # We need to provide initial guess for all 9 slots (fixed ones are ignored)
    initial = [0.0, 0.0] + list(current_joints) + [0.0]

    # Solve IK
    solution = UR3E_CHAIN.inverse_kinematics(
        target_position=target_xyz,
        initial_position=initial
    )

    # Extract only the 6 real joint angles (indices 2-7)
    return list(solution[2:8])

class RRTFNPlannerNode(Node):
    def __init__(self):
        super().__init__('rrtfn_planner')
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.goal_sub = self.create_subscription(Point, '/hand_goal', self.goal_callback, 10)

        self.current_config = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
        self.trajectory = []
        self.traj_index = 0
        self.last_goal_xyz = None
        self.replan_threshold = 0.08  # 8cm hand movement triggers replan
        self.is_planning = False

        self.get_logger().info('RRT*FN Planner with real IK started!')
        self.get_logger().info('Waiting for hand goals on /hand_goal ...')

        self.publish_timer = self.create_timer(0.02, self.publish_joint_state)

    def publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        if self.trajectory and self.traj_index < len(self.trajectory):
            self.current_config = self.trajectory[self.traj_index]
            self.traj_index += 1
        msg.position = self.current_config
        self.joint_pub.publish(msg)

    def goal_callback(self, msg):
        if self.is_planning:
            return

        # Let current path mostly finish before replanning
        if self.trajectory and self.traj_index < len(self.trajectory) * 0.7:
            return

        target_xyz = [msg.x, msg.y, msg.z]

        # Only replan if hand moved meaningfully
        if self.last_goal_xyz is not None:
            d = np.linalg.norm(np.array(target_xyz) - np.array(self.last_goal_xyz))
            if d < self.replan_threshold:
                return

        self.last_goal_xyz = target_xyz
        self.is_planning = True

        # Solve IK to convert Cartesian target into joint-space goal
        try:
            goal_config = solve_ik(target_xyz, self.current_config)
        except Exception as e:
            self.get_logger().warn(f'IK failed: {e}')
            self.is_planning = False
            return

        # Verify IK accuracy — where will the end-effector actually be?
        initial = [0.0, 0.0] + list(goal_config) + [0.0]
        achieved = UR3E_CHAIN.forward_kinematics(initial)[:3, 3]
        err = np.linalg.norm(np.array(target_xyz) - achieved)

        self.get_logger().info(
            f'Target: ({msg.x:+.2f},{msg.y:+.2f},{msg.z:+.2f}) '
            f'-> achieved: ({achieved[0]:+.2f},{achieved[1]:+.2f},{achieved[2]:+.2f}) '
            f'err: {err:.3f}m'
        )

        # Plan path from current joint config to IK solution
        path = rrtfn(self.current_config, goal_config, max_nodes=300, max_iter=1500)

        if path is None:
            self.get_logger().warn('No path found.')
            self.is_planning = False
            return

        self.trajectory = interpolate_path(path, points_per_segment=30)
        self.traj_index = 0
        duration_s = len(self.trajectory) * 0.02
        self.get_logger().info(f'Path: {len(path)} waypoints, {duration_s:.1f}s motion')
        self.is_planning = False

def main(args=None):
    rclpy.init(args=args)
    node = RRTFNPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass

if __name__ == '__main__':
    main()
