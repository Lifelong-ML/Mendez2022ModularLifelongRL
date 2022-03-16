import numpy as np
from spinup.utils.mpi_tools import mpi_sum, num_procs
from gym import spaces
from gym.core import Env
from robosuite.wrappers import Wrapper
from collections import OrderedDict
from copy import deepcopy

class RunningMeanStd(object):
    def __init__(self, epsilon= 1e-4, shape=None):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x) -> None:
        # batch_mean = np.mean(arr, axis=0)
        # batch_var = np.var(arr, axis=0)
        # batch_count = arr.shape[0]
        batch_count = num_procs()
        batch_sum = mpi_sum(x)    # sync envs across MPI processes and get mean and std  
        batch_mean = batch_sum / batch_count

        batch_sum_sq = mpi_sum((x - batch_mean) ** 2)
        batch_var = batch_sum_sq / batch_count 

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class NormalizedMPIWrapper(Wrapper, Env):
    def __init__(
        self,
        env,
        clip_obs=10.,
        training=True
    ): 
        super().__init__(env=env)
        self.training = training
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        if isinstance(self.observation_space, spaces.Dict):
            self.obs_keys = set(self.observation_space.spaces.keys())
            self.obs_spaces = self.observation_space.spaces
            self.obs_rms = {key: RunningMeanStd(shape=space.shape) for key, space in self.obs_spaces.items()}
        else:
            self.obs_keys, self.obs_spaces = None, None
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)

        self.clip_obs = clip_obs
        self.epsilon = 1e-8

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.training:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                self.obs_rms.update(obs)
        obs = self.normalize_obs(obs)

        return obs, reward, done, info

    def reset(self):
        """
        Reset all environments
        :return: first observation of the episode
        """
        obs = self.env.reset()
        return self.normalize_obs(obs)

    def normalize_obs(self, obs):
        obs_ = deepcopy(obs)    # avoid modifying original object
        if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
            for key in self.obs_rms.keys():
                obs_[key] = self._normalize_obs(obs[key], self.obs_rms[key]).astype(np.float32)
        else:
            obs_ = self._normalize_obs(obs, self.obs_rms).astype(np.float32)
        return obs_

    def _normalize_obs(self, obs, obs_rms):
        """
        Helper to normalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: normalized observation
        """
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
