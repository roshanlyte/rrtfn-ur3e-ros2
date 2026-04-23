import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import cv2
import mediapipe as mp
from collections import deque
import math

class ArmTrackerNode(Node):
    def __init__(self):
        super().__init__('arm_tracker')
        self.publisher_ = self.create_publisher(Point, '/hand_goal', 10)
        self.get_logger().info('Arm Tracker Node started!')

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Cannot open camera')
            return

        self.x_buf = deque(maxlen=5)
        self.y_buf = deque(maxlen=5)
        self.d_buf = deque(maxlen=5)

        self.timer_cam = self.create_timer(0.033, self.process_frame)
        self.timer_pub = self.create_timer(0.2, self.publish_goal)
        self.latest_goal = None

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

            # Landmarks: 12=right shoulder, 14=right elbow, 16=right wrist
            shoulder = results.pose_landmarks.landmark[12]
            elbow    = results.pose_landmarks.landmark[14]
            wrist    = results.pose_landmarks.landmark[16]

            # Depth proxy: distance from shoulder to wrist (in image)
            # When you reach forward, shoulder-wrist distance shrinks
            # When your arm is to the side, it's larger
            dx = wrist.x - shoulder.x
            dy = wrist.y - shoulder.y
            arm_extension = math.sqrt(dx*dx + dy*dy)  # 0..~0.5 in normalised coords

            self.x_buf.append(wrist.x)
            self.y_buf.append(wrist.y)
            self.d_buf.append(arm_extension)

            sx = sum(self.x_buf) / len(self.x_buf)
            sy = sum(self.y_buf) / len(self.y_buf)
            sd = sum(self.d_buf) / len(self.d_buf)

            # --- ROBOT WORKSPACE MAPPING ---
            # UR3e base sits at origin, z-axis is UP, x-axis is FORWARD
            # Target workspace: a nice box 30-50cm in front of the robot
            #
            # Camera (after flip)      -> Robot frame
            #   wrist x (0 left, 1 right)  -> robot Y (side-to-side)
            #   wrist y (0 top,  1 bottom) -> robot Z (up-down)  inverted
            #   arm extension              -> robot X (forward)

            goal = Point()
            # x = forward distance from base: 0.25 (close) to 0.45m (far)
            # arm_extension ranges ~0.1 (arm tucked) to 0.4 (fully out)
            goal.x = 0.25 + min(max(sd, 0.1), 0.4) * 0.5
            # y = side: left/right symmetric around 0
            goal.y = (sx - 0.5) * 0.5   # -0.25 to +0.25 m
            # z = height: raise hand = robot goes up
            goal.z = 0.5 - (sy * 0.4)   # 0.1 (low) to 0.5 (high)

            self.latest_goal = goal

            # Visual feedback
            h, w, _ = frame.shape
            wx, wy = int(wrist.x * w), int(wrist.y * h)
            cv2.circle(frame, (wx, wy), 25, (0, 255, 0), -1)

            # Draw arm extension bar
            bar_w = int(sd * 400)
            cv2.rectangle(frame, (10, h - 40), (10 + bar_w, h - 20), (0, 255, 255), -1)
            cv2.putText(frame, 'Arm extension (= robot forward reach)',
                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.putText(frame,
                f'Robot target  x={goal.x:.2f}  y={goal.y:+.2f}  z={goal.z:.2f}  (metres)',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame,
                'Move arm shown on RIGHT of mirror.  Extend arm to reach further.',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow('Arm Tracker (ROS2)', frame)
        cv2.waitKey(1)

    def publish_goal(self):
        if self.latest_goal is not None:
            self.publisher_.publish(self.latest_goal)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArmTrackerNode()
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
