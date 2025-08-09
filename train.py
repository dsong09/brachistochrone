import numpy as np
from path_generator import PathGenerator, START_POINT, END_POINT, NUM_CONTROL_POINTS
from brachistochrone_env import BrachistochroneEnv
from agent import Agent

def run_episode(env, agent):
    # reset environment and agent
    agent.reset()
    obs = env.reset()

    done = False
    total_reward = 0.0

    while not done:
        # agent outputs y-control points (excluding start/endpoints)
        action = agent.action()
        obs, reward, done = env.step(action)
        total_reward += reward

    agent.update_after_episode(total_reward)
    return total_reward

if __name__ == "__main__":
    path = PathGenerator(START_POINT, END_POINT, NUM_CONTROL_POINTS)
    env = BrachistochroneEnv(path)
    agent = Agent(NUM_CONTROL_POINTS)

    reward = run_episode(env, agent)
    print(f"Episode finished. Total reward: {reward:.4f}")
