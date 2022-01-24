import numpy as np
import rospy
from gazebo_msgs.srv import GetModelState, SpawnModel, DeleteModel
from geometry_msgs.msg import *
import tf
from std_srvs.srv import Empty

pillow = {'init' : {'x' : 0.38, 'y': -0.05, 'z' : 0.19, 'row' : 0, 'pitch'  : 0, 'yaw' : 1.57},
          'lower_lim': {'x' : 0.25, 'y': -0.14, 'z' : 0.19, 'row' : 0, 'pitch' : 0, 'yaw' : -1.57},
          'upper_lim': {'x' : 0.38, 'y': 0.2, 'z' : 0.19, 'row' : 0, 'pitch' : 0, 'yaw' : 1.57}
          }

goal = {'init' : {'x' : 0.4, 'y': 0.2, 'z' : 0.12, 'row' : 0, 'pitch'  : 0, 'yaw' : 0},
        'lower_lim': {'x' : 0.25, 'y': -0.14, 'z' : 0.19, 'row' : 0, 'pitch' : 0, 'yaw' : -1.57},
        'upper_lim': {'x' : 0.38, 'y': 0.2, 'z' : 0.19, 'row' : 0, 'pitch' : 0, 'yaw' : 1.57}
        }

class World:
    def __init__(self):
        rospy.sleep(1)
        self.available_model = ["pillow", "goal", "Bed", "BedFrame", "niryo_one"]
        self.pillow_pose = self.get_model_state("pillow")
        self.goal_pose = self.get_model_state("goal")
        self.bed_pose = self.get_model_state("Bed")
        self.bedframe_pose = self.get_model_state("BedFrame")

    def update_world_state(self):
        self.pillow_pose = self.get_model_state("pillow")
        self.goal_pose = self.get_model_state("goal")
        self.bed_pose = self.get_model_state("Bed")
        self.bedframe_pose = self.get_model_state("BedFrame")

    def reset(self, random = False):
        # remove bed and pillow and respawn them
        rospy.wait_for_service('/gazebo/reset_world')
        
        self.delete_model("pillow")
        self.delete_model("goal")
        self.spawn("pillow", random=random)
        self.spawn("goal", random=random)
        self.update_world_state()
           
        try:
            self.reset_gazebo_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        # self.reset_gazebo_world()

    def check_bed_movement(self):
        ''' TODO: find a way to get difference between two Pose()
        
        compare new location of bed and bedframe, if move beyond
        certain length then reset the environment (ie return done as True)
        use inside step funtion to return correct done condition
        '''
        if self.bedframe_pose != self.get_model_state("BedFrame"):
                    print("Bed Frame has moved")
                    print(self.bedframe_pose)
                    print(self.get_model_state("BedFrame"))
                    print(self.bedframe_pose)
                    return True

        if self.bed_pose != self.get_model_state("Bed"):
            print("Bed has moved")
            return True
        
        return False

    def pillow_move(self):
        # if pillow doesn't move, return 0
        # if pillow move, calculate reward
        new_pose = self.get_model_state("pillow")
        if new_pose == self.pillow_pose:
            return False

        else:
            self.pillow_pose = new_pose
            return True

    def pillow_move_up(self):
        if np.abs(self.get_height("pillow") - self.pillow_z) > 0.01:
            return True
        return False
    
    def get_model_state(self, model):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        except rospy.ServiceException as e:
            print("Error")
            print("Service call failed: %s"%e)
        state = get_state(model, '')
        if state.success == False:
            print("Retrying get_model_state: ", model)
            try:
                get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            except rospy.ServiceException as e:
                print("Error")
                print("Service call failed: %s"%e)
            print("Error getting model state, Available model are:")
            print(self.available_model)
        state = get_state(model, '')
        return state

    def get_model_height(self, model):
        state = self.get_model_state(model)
        # print(state.pose.position.z)
        return state.pose.position.z

    def spawn(self, model, x=None, y=None, z=None, row=None, pitch=None, yaw=None, random=False):
        '''
        Spawn SDF Model

        Args: 
            model ('pillow' or 'goal'): model that want to spawn wrt to world
                pillow and goal locations are limit to x = [0.25 to 0.4] and y = [-0.15 to 0.2] to keep it within camera and arm workspace 
                z should be 0.19 for pillow and 0.12 for goal
            x, y, z row, pitch, yaw : coordinate and rotation with repect to the world coordinate
            
        Returns:
            None
        '''
        
        if model.lower() == "pillow":
            f = open('/home/joker/Niryo/src/niryo_one_ros_simulation/niryo_one_gazebo/models/pillow/model.sdf','r')
            model_data = pillow
        elif model.lower() == "goal":
            f = open('/home/joker/Niryo/src/niryo_one_ros_simulation/niryo_one_gazebo/models/goal/model.sdf','r')
            model_data = goal
        if x==None:
            x = model_data['init']['x']
        if y==None:
            y = model_data['init']['y']
        if z==None:
            z = model_data['init']['z']
        if row==None:
            row = model_data['init']['row']
        if pitch==None:
            pitch = model_data['init']['pitch']
        if yaw==None:
            yaw = model_data['init']['yaw']
        
        if random==True:
            x = np.random.uniform(model_data['lower_lim']['x'], model_data['upper_lim']['x'])
            y = np.random.uniform(model_data['lower_lim']['y'], model_data['upper_lim']['y'])
            z = np.random.uniform(model_data['lower_lim']['z'], model_data['upper_lim']['z'])
            row = np.random.uniform(model_data['lower_lim']['row'], model_data['upper_lim']['row'])
            pitch = np.random.uniform(model_data['lower_lim']['pitch'], model_data['upper_lim']['pitch'])
            yaw = np.random.uniform(model_data['lower_lim']['yaw'], model_data['upper_lim']['yaw'])
            # TODO Add randomize goal and pillow spawning

        initial_pose = Pose()
        q = tf.transformations.quaternion_from_euler(row, pitch, yaw)
        initial_pose.position.x = x
        initial_pose.position.y = y
        initial_pose.position.z = z
        initial_pose.orientation.x = q[0]
        initial_pose.orientation.y = q[1]
        initial_pose.orientation.z = q[2]
        initial_pose.orientation.w = q[3]
        sdff = f.read()
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_prox(model.lower(), sdff, "", initial_pose, "world")
    
    def delete_model(self, model):
        print("deleting")
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            resp_delete = delete_model(model.lower())
        except rospy.ServiceException as e:
            print("Delete Model service call failed: {0}")

if __name__ == '__main__':
    print("Inside world.py")
    world = World()
    # world.reset()
    # print(world.get_model_state("niryo_one"))
    # world.spawn("Pillow", 0.38, 0.2, 0.19, 0, 0, 0)
    # world.spawn("Pillow", random=True)
    # world.spawn("goal", random=True)
    # raw_input()
    # world.delete_model("Pillow")
    # world.delete_model("goal")
    # world.check_bed_movement()
    world.reset()