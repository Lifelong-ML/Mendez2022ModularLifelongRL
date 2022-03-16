from .minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
from .wrappers import ReseedWrapper, FullyObsWrapper
import gym
from .comp_envs.few_shot import FewShotCompEnv
import numpy as np

# MOD_OBJECT_TO_IDX = {
#     'unseen'        : 0,
#     'empty'         : 0,
#     'wall'          : 2,
#     'floor'         : 1,
#     'door'          : 3,
#     'key'           : 0,
#     'ball'          : 4,
#     'box'           : 0,
#     'goal'          : 0,
#     'lava'          : 1,
#     'agent'         : 0,
#     'food'          : 1,
#     'heat'          : 1,
#     'target'        : 5
# }

STATIC_OBJECTS = set([
    OBJECT_TO_IDX['wall'],
    OBJECT_TO_IDX['floor'],
    OBJECT_TO_IDX['lava'],
    OBJECT_TO_IDX['food'],
    OBJECT_TO_IDX['heat']
])

class FewShotWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to enable few-shot learning of compositional tasks
    without permitting zero-shot transfer. Hides details about
    how tasks are different (static object descriptions, goal
    information, target object, and agent variation).
    """

    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        grid = obs['image']
        for obj_type, obj_idx in OBJECT_TO_IDX.items():
            bool_idx = (grid[:, :, 0] == obj_idx)
            grid[bool_idx, 0] = MOD_OBJECT_TO_IDX[obj_type]
            if obj_idx in STATIC_OBJECTS:
                grid[bool_idx, 1] = 0
        return obs

class ChanneledWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = self.observation_space.spaces['image'].shape
        self.observation_space.spaces['image'] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[0], shape[1], 7),  # number of cells: wall, floor, door, lava, food, target, agent
            dtype='uint8')
        # exit()

    def observation(self, obs):
        objects_of_interest = ['wall', 'floor', 'door', 'food', 'lava', 'target', 'agent']
        grid = obs['image']
        grid_channeled = np.zeros((grid.shape[0], grid.shape[1], len(objects_of_interest)))
        for i, obj_name in enumerate(objects_of_interest):
            obj_idx = OBJECT_TO_IDX[obj_name]
            bool_idx = (grid[:, :, 0] == obj_idx)
            if obj_idx == OBJECT_TO_IDX['target']:
                grid_channeled[bool_idx, i] = grid[bool_idx, 1] + 1     # ball color
            elif obj_idx == OBJECT_TO_IDX['door']:
                grid_channeled[bool_idx, i] = grid[bool_idx, 2]     # door state
            elif obj_idx == OBJECT_TO_IDX['agent']:
                if isinstance(self.env, FullyObsWrapper):
                    grid_channeled[bool_idx, i] = grid[bool_idx, 2] + 1     # agent state
                else:
                    grid_channeled[grid.shape[0] // 2, grid.shape[1] // 2, i] = self.env.agent_dir + 1
            else:
                grid_channeled[bool_idx, i] = 1
        obs['image'] = grid_channeled
        return obs