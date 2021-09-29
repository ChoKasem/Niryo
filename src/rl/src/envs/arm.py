import rospy
from cv_bridge import CvBridge, CvBridgeError
from control_msgs.msg import GripperCommandActionGoal
from sensor_msgs.msg import JointState, Image

from std_msgs.msg import String
from gazebo_msgs.srv import GetModelState, SpawnModel
from geometry_msgs.msg import *
from std_srvs.srv import Empty
import matplotlib.pyplot as plt
import cv2


import niryo_moveit_commander

class Arm:
    def __init__(self):
        '''
        0 = joint_1
        1 = joint_2
        2 = joint_3
        3 = joint_4
        4 = joint_5
        5 = joint_6
        6 = tool_joint // cannot be control here, just for connecting with tools
        '''
        self.image = None
        self.depth = None
        self.joint_angle = None
        self.bridge = CvBridge()
        self.command = niryo_moveit_commander.MoveGroupPythonInteface("arm")
        
        # rospy.init_node('Niryo_State', anonymous=True)
        rospy.loginfo("Initializing State Subscriber")
        
        rospy.Subscriber('/joint_states', JointState, self.joint_cb)
        # rospy.Subscriber('/')
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_cb)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_cb)
        rospy.sleep(1)

    def get_state_callback(self, data):
        self.joint_angle

    def image_cb(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
    def depth_cb(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def joint_cb(self, msg):
        self.joint_angle = msg

    def get_end_effector_pose(self):
        return self.command.get_pose()

    def test_arm(self):
        print("Success import!")
        
class Gripper:
    
    def __init__(self):
        '''
        0 = tool_joint
        1 = mainsupport_joint
        2 = gripper_joint
        3 = left_clamp_joint
        4 = left_rod_joint
        5 = motor_joint
        6 = right_gear_joint
        7 = right_clamp_joint
        8 = right_rod_joint
        '''
        self.gripper_angle = 0
        self.gripper_pub = rospy.Publisher('/gripper_controller/gripper_cmd/goal', GripperCommandActionGoal, queue_size=10)
        rospy.sleep(1)

    def grab_angle(self, angle): #angle max at 1.2
        if angle > 1.2:
            angle = 1.2
        gripperGoal = GripperCommandActionGoal()
        gripperGoal.goal.command.position = angle
        print(gripperGoal)
        rospy.sleep(1)
        self.gripper_angle = angle
        self.gripper_pub.publish(gripperGoal)

if __name__ == '__main__':
    print("Inside arm.py")
    rospy.loginfo("Initialize Niryo RL Node")
    rospy.init_node('Arm_Test_Node',
                    anonymous=True)
    arm = Arm()
    print("Moving Arm")
    arm.command.go_to_pose_goal(0.350840341432, -0.058138712168, 0.276432223498, 0.50174247142, 0.501506407284, 0.498433947182, 0.498306548344)
    print("Printing Arm Joint Angle")
    print(arm.joint_angle)
    print("Printing end effector pose")
    print(arm.get_end_effector_pose())