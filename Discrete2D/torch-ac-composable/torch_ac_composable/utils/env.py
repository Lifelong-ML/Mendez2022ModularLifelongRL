import gym
import gym_minigrid


def make_env(env_key, seed=1337):
    env = gym.make(env_key)
    env.seed(seed)
    return env
