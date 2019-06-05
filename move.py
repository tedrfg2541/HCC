import sys
import move_arm.srv 
import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose
from math import pi
from std_msgs.msg import String, Float64
from moveit_commander.conversions import pose_to_list
import ik_4dof

def handle_move_arm(req):
# initial publisher for gripper command topic which is used for gripper control
	pub_gripper = rospy.Publisher("/gripper_joint/command",Float64,queue_size=1)

	check = True
	# you need to check if moveit server is open or not
	while(check):
		check = False
		try:
			# Instantiate a MoveGroupCommander object. This object is an interface to a planning group 
			# In our case, we have "arm" and "gripper" group

			move_group = moveit_commander.MoveGroupCommander("arm")
		except:
			print "moveit server isn't open yet"
			check = True


	# First initialize moveit_commander
	moveit_commander.roscpp_initialize(sys.argv)

	# Instantiate a RobotCommander object. 
	# Provides information such as the robot's kinematic model and the robot's current joint states
	robot = moveit_commander.RobotCommander()


	# We can get the name of the reference frame for this robot:
	planning_frame = move_group.get_planning_frame()
	print "============ Planning frame: %s" % planning_frame

	# We can also print the name of the end-effector link for this group:
	eef_link = move_group.get_end_effector_link()
	print "============ End effector link: %s" % eef_link

	# We can get a list of all the groups in the robot:
	group_names = robot.get_group_names()
	print "============ Available Planning Groups:", group_names

	# Sometimes for debugging it is useful to print the entire state of the
	# robot:
	print "============ Printing robot state", robot.get_current_state()
	print ""
		
	### Go home
	home() 

	############################ Method : Using IK to calculate joint value ############################

	# After determining a specific point where arm should move, we input x,y,z,degree to calculate joint value for each wrist. 

	pose_goal = Pose()
	pose_goal.position.x = req.x
	pose_goal.position.y = req.y
	pose_goal.position.z = 0.042
	# ik_4dof.ik_solver(x, y, z, degree)
	joint_value = ik_4dof.ik_solver(pose_goal.position.x, pose_goal.position.y, pose_goal.position.z, -90)

	for joint in joint_value:
		joint = list(joint)
		# determine gripper state
		joint.append(0)
		try:
			move_group.go(joint, wait=True)
		except:
			rospy.loginfo(str(joint) + " isn't a valid configuration.")

	# Reference: https://ros-planning.github.io/moveit_tutorials/doc/move_group_python_interface/move_group_python_interface_tutorial.html

	### close gripper

	rospy.sleep(2)
	grip_data = Float64()
	grip_data.data = 2.0 
	pub_gripper.publish(grip_data)

	### open gripper

	rospy.sleep(2)
	grip_data = Float64()
	grip_data.data = 0.5
	pub_gripper.publish(grip_data)
	rospy.sleep(2)

	### Go home
	home() 
		


def home():

	############################ Method : Joint values (Go home!)############################


	# We can get the joint values from the group and adjust some of the values:

	# Go home!!!
    move_group = moveit_commander.MoveGroupCommander("arm")
	joint_goal = move_group.get_current_joint_values()
	joint_goal[0] = 0   		# arm_shoulder_pan_joint
	joint_goal[1] = -pi*5/13   	# arm_shoulder_lift_joint
	joint_goal[2] = pi*3/4   	# arm_elbow_flex_joint
	joint_goal[3] = pi/3   		# arm_wrist_flex_joint
	joint_goal[4] = 0   		# gripper_joint

	# The go command can be called with joint values, poses, or without any
	# parameters if you have already set the pose or joint target for the group
	move_group.go(joint_goal, wait=True)

	# Calling ``stop()`` ensures that there is no residual movement
    move_group.stop()

def move_arm_server():
    rospy.init_node('move_serve')
    s = rospy.Service('arm_planning', move_arm, handle_move_arm)
    print "Ready to get x and y."
    rospy.spin()

if __name__ == "__main__":
    move_arm_server()
    