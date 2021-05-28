from rl import PPO
from rl_base import get_sample_state, Memory
from agent import ActorCritic

NUM_OUTPUTS = 6+1

def sample_ppo_update():
    def mock_step(action):
        state = get_sample_state(img_wh[0], img_wh[1])
        return state, 1, False, {}

    # Initializing the Actor Critic
    img_wh = [512, 512]
    agent = ActorCritic(
        rgb_img_shape=(img_wh[0], img_wh[1], 3),
        depth_img_shape=(img_wh[0], img_wh[1], 1),
        num_joints=6,
        pillow_pose_size=3,
        num_outputs=NUM_OUTPUTS
    )
    agent_old = ActorCritic(
        rgb_img_shape=(img_wh[0], img_wh[1], 3),
        depth_img_shape=(img_wh[0], img_wh[1], 1),
        num_joints=6,
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
        state = get_sample_state(img_wh[0], img_wh[1])
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
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
                _, value = agent(state.to_tensor())
                print("Value after update" ,value)

            if done:
                break

def sample_forward_pass(img_width, img_height):
    from niryo_env import Env # TODO: Edit this

    # Initializing the Actor Critic
    img_wh = [512, 512]
    agent = ActorCritic(
        rgb_img_shape=(img_width, img_height, 3),
        depth_img_shape=(img_width, img_height, 1),
        num_joints=6,
        pillow_pose_size=3,
        num_outputs=NUM_OUTPUTS
    )
    
    # Initialize environment
    env = Env() # TODO: Edit this

    # Step
    sample_action = None # TODO: Edit this
    state = env.reset()
    state, reward, done, info = env.step(sample_action)

    # Forward pass
    action_dist, value = agent(state.to_tensor())
    print (action_dist, value)
    
    

"""TODO: 
- What if multiple pillow in image
"""
if __name__ == '__main__':
    sample_ppo_update()



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
