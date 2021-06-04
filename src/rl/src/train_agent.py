import numpy as np
import cv2

from rl import PPO
from rl_base import get_sample_state, Memory
from agent import ActorCritic
from niryo_env import Niryo

NUM_OUTPUTS = Niryo.action_dim

def sample_ppo_update():
    def mock_step(action):
        state = get_sample_state(img_wh[0], img_wh[1], num_joints)
        return state, 1, False, {}

    # Initializing the Actor Critic
    img_wh = [480, 640]
    num_joints = 12
    agent = ActorCritic(
        rgb_img_shape=(img_wh[0], img_wh[1], 3),
        depth_img_shape=(img_wh[0], img_wh[1], 1),
        num_joints=num_joints,
        pillow_pose_size=3,
        num_outputs=NUM_OUTPUTS
    )
    agent_old = ActorCritic(
        rgb_img_shape=(img_wh[0], img_wh[1], 3),
        depth_img_shape=(img_wh[0], img_wh[1], 1),
        num_joints=num_joints,
        pillow_pose_size=3,
        num_outputs=NUM_OUTPUTS
    )
    ppo = PPO(agent, agent_old, buffer_size=1, mini_batch_size=1)
    memory = Memory()
    max_episodes = 1
    max_timesteps = 3
    update_timestep = 2

    time_step = 0
    for i_episode in range(1, max_episodes+1):
        state = get_sample_state(img_wh[0], img_wh[1], num_joints)
        for t in range(max_timesteps):
            time_step += 1

            # Get action from agent and perform a step
            action_dist, _ = agent(state.to_tensor())
            action = action_dist.sample()
            state, reward, done, _ = mock_step(action)

            # Saving the transition
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_dist.log_prob(action))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                _, value = agent(state.to_tensor())
                print("Value before update" , value)
                # ppo.update(memory)
                memory.clear_memory()
                time_step = 0
                _, value = agent(state.to_tensor())
                print("Value after update" ,value)

            if done:
                break
    print("RGB Info", state.rgb.shape, np.mean(state.rgb))
    print("Depth Info", state.depth.shape, np.mean(state.depth))
    print("Joint Info", state.joint.shape)

def sample_forward_pass(img_width, img_height):
    # Initialize environment
    env = Niryo()

    # Initializing the Actor Critic
    agent = ActorCritic(
        rgb_img_shape=(img_width, img_height, 3),
        depth_img_shape=(img_width, img_height, 1),
        num_joints=env.num_joints,
        pillow_pose_size=3,
        num_outputs=env.action_dim
    )

    # Step
    sample_action =  [0,0,-1,0,0,0,0]
    state = env.reset_pose()
    state, reward, done, info = env.one_hot_step(sample_action)

    # cv2.imshow("image", state.rgb)
    # cv2.imshow("depth", state.depth)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("RGB Info", state.rgb.shape, np.mean(state.rgb))
    print("Depth Info", state.depth.shape, np.mean(state.depth))
    print("Joint Info", state.joint.shape)

    # TODO: Depth image is currently lots of NaNs
    # Add a dummy depth image for now
    state.depth = np.random.uniform(0, 1, (img_width, img_height, 1))

    # Add a dummy pillow pose
    state.pillow = np.random.uniform(0, 100, (3))

    # TODO: Have to convert from action to one-hot
    # TODO: Current setup (i.e. agent+env) doesn't allow for negatives in the one-hot
    # Forward pass
    action_dist, value = agent(state.to_tensor())
    print (action_dist, value)
    print(action_dist.sample())
    
    
    

"""TODO: 
- What if multiple pillow in image
"""
if __name__ == '__main__':
    # sample_ppo_update()
    sample_forward_pass(480, 640)


# Pseudocode for usage
# import env, action_space from niryo

# for _ in episodes
#     state = env.reset()
#     for _ in steps:
#         action = agent(state)
#         state, reward, done, info = env.step(action)
#         agent.update()

# action_space= {UP, DOWN, ...increments}
# move_to_pose()
