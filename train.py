import numpy as np
from path_generator import PathGenerator, START_POINT, END_POINT
from brachistochrone_env import BrachistochroneEnv
from agent import Agent
from replay_buffer import ReplayBuffer

def run_episode(env, agent, replay_buffer):
    # reset environment and agent
    agent.reset()
    obs = env.reset()

    done = False
    total_reward = 0.0

    while not done:
        # state from observation
        state = np.concatenate([
            np.array(obs['position']),
            np.array([obs['velocity']]),
            obs['y_control']
        ])

        # agent chooses action based on current state (to be implemented)
        action = agent.action()

        # step
        next_obs, reward, done = env.step(action)

        next_state = np.concatenate([
            np.array(next_obs['position']),
            np.array([next_obs['velocity']]),
            next_obs['y_control']
        ])

        # store experience
        replay_buffer.add(state, action, reward, next_state, done)

        obs = next_obs
        total_reward += reward

    # sample batch if buffer is large enough
    if replay_buffer.size() > replay_buffer.batch_size:
        batch = replay_buffer.sample()
        agent.train_on_batch(*batch)

    agent.update_after_episode(total_reward)
    return total_reward

if __name__ == "__main__":
    path = PathGenerator(START_POINT, END_POINT)
    env = BrachistochroneEnv(path)
    agent = Agent(y_min=-1.0, y_max=6.0)

    buffer_size = 100000
    batch_size = 64
    replay_buffer = ReplayBuffer(max_size=buffer_size, batch_size=batch_size)

    reward = run_episode(env, agent)
    print(f"Episode finished. Total reward: {reward:.4f}")
