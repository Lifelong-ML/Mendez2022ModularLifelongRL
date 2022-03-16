from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from itertools import permutations

class FewShotCompEnv(MiniGridEnv):
    """
    Compositional environment with different 
    agents, obstacles, and target objects. 
    Exploration is required since descriptions 
    for zero-shot performance are hidden
    """

    def __init__(
        self, 
        size=8,
        agent_index=0,
        env_object=None,
        target='red',
        seed_offset=0,
        num_distractor_colors=0
    ):
        self.agent_index = agent_index
        self.action_permutation = self._idx_to_permutation(agent_index)
        self.env_object = env_object
        self.target_type = 'target'
        self.target_color = target
        self.seed_offset = seed_offset
        self.num_distractor_colors = num_distractor_colors
        super().__init__(
            grid_size=size,
            max_steps=size*size,
        )
        

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Add vertical "wall" of obstacle type
        if self.env_object == 'wall':
            # Cannot walk through
            obstacle_type = Wall
        elif self.env_object == 'lava':
            # Die if try to walk through
            obstacle_type = Lava 
        elif self.env_object == 'floor':
            # Does nothing
            obstacle_type = Floor
        elif self.env_object == 'food':
            # Positive if try to walk through
            obstacle_type = Food
        else:
            raise ValueError('Add a valid object type')

        # Add door for walls and gaps for others
        split_idx = self._rand_int(2, width - 2)
        self.grid.vert_wall(split_idx, 1, height - 2, obj_type=obstacle_type)
        door_or_gap_idx = self._rand_int(1, width - 2)
        if self.env_object == 'wall':
            self.put_obj(Door('yellow', is_locked=False), split_idx, door_or_gap_idx)
        else:
            self.grid.set(split_idx, door_or_gap_idx, None)

        self.place_agent()

        self.place_obj(Target(self.target_color))  # always place target colored ball
        distractor_colors = set(['red', 'green', 'blue', 'purple'])#, 'yellow'])
        distractor_colors.remove(self.target_color)     # add from colors not target
        distractor_colors = self._rand_subset(distractor_colors, self.num_distractor_colors)
        for color in distractor_colors:
            self.place_obj(Target(color))

        # Create mission string
        self.mission = "Go to {} target interacting\nwith {} with agent {}".format(self.target_color, self.env_object, self.agent_index)# self.action_permutation)

    def step(self, action):
        """ Permute the action before taking env step """
        
        preCarrying = self.carrying
        action = self.action_permutation[action]
        obs, reward, done, _ = super().step(action)
        assert reward == 0, "MiniGridEnv should always give 0 reward"
        
        curr_pos = self.agent_pos
        curr_cell = self.grid.get(*curr_pos)

        if curr_cell is not None:
            if curr_cell.type == self.target_type and curr_cell.color == self.target_color:
                reward = self._reward()
                done = True
            elif curr_cell.type == 'lava':
                reward += self._obstacle_reward('lava')
            # If pickup and food, reward
            elif action == self.actions.pickup and curr_cell.type == 'food':
                reward += self._obstacle_reward('food')
                self.grid.set(*curr_pos, None)

        return obs, reward, done, {}

    def seed(self, seed=1337):
        return super().seed(seed + self.seed_offset)


    def _obstacle_reward(self, obs_type):
        """ Compute rewards associated to object """
        r = 0
        if obs_type == 'lava':
            r = -0.05
        elif obs_type == 'food':
            r = 0.05
        return r

    def _idx_to_permutation(self, agent_index):
        """ Compute the permutation corresponding to an agent index """
        actions_to_permute = [0, 1, 2, 5]
        permuted_actions = list(list(permutations(actions_to_permute))[agent_index])
        return permuted_actions[:3] + [3, 4] + permuted_actions[-1:] + [6]

class Food(WorldObj):
    def __init__(self):
        super().__init__('food', 'blue')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (0, 128, 255)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

class Target(WorldObj):
    def __init__(self, color='green'):
        super().__init__('target', color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

for size in [5, 6, 7, 8]:
    i = 0
    for agent_index in range(24):
        for env_object in [None, 'wall', 'floor', 'lava', 'food']:
            for target in ['red', 'green', 'blue', 'purple']:
                env_id = 'MiniGrid-FewShot-{}-{}-{}-{}x{}-v0'.format(agent_index, env_object, target, size, size)
                register(
                    id=env_id,
                    entry_point='gym_minigrid.comp_envs:FewShotCompEnv',
                    kwargs={
                        'size'                      : size,
                        'agent_index'               : agent_index,
                        'env_object'                : env_object,
                        'target'                    : target,
                        'seed_offset'               : i,
                        'num_distractor_colors'     : 3
                    }
                )
                i += 1
