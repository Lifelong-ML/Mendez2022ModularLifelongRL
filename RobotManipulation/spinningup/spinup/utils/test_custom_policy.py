from spinup.utils.test_policy import run_policy#, load_policy_and_env
import os
import os.path as osp
import torch
import argparse

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers.gym_wrapper_notflat import GymWrapper

import spinup.algos.pytorch.ppo.core as core
from spinup.algos.pytorch.ppo.ppo import ppo
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.mpi_tools import mpi_fork

def create_env(args):
    # options = {}
    # options["env_name"] = "CompositionalEnv"
    # options["robots"] = choose_robots(exclude_bimanual=True)
    # options["task"] = choose_task()
    # options["object_type"] = choose_object().lower()
    # options["obstacle"] = choose_obstacle()
    # options["controller_configs"] = choose_controller()
    options = {}
    possible_robots = ["IIWA", "Kinova3", "Panda", "Sawyer"]
    possible_obstacles = ["None", "ObjectDoor", "ObjectWall"]
    possible_objects = ["can", "milk", "bread", "cereal"]

    options["robots"] = possible_robots[args.robot]
    options["obstacle"] = possible_obstacles[args.obstacle]
    options["object_type"] = possible_objects[args.object]
    options["env_name"] = "CompositionalEnv"
    options["task"] = "Lift"

    for key in options.keys():
        print(key, options[key])

    if options["obstacle"] == "None":
        options["obstacle"] = None
    controller_name = "OSC_POSITION"
    # controller_name = "JOINT_POSITION"
    # controller_name = "OSC_POSE"
    options["controller_configs"] = load_controller_config(
        default_controller=controller_name)

    print()
    print("Press \"H\" to show the viewer control panel.")

    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        reward_shaping=True,
        ignore_done=True,
        use_camera_obs=False,
        use_goal_obs=True,
        control_freq=20,
        horizon=500
    )

    env.reset()

    return GymWrapper(env)

def load_policy(fpath, itr='last', deterministic=False, task_id=0):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action = load_pytorch_policy(fpath, itr, deterministic, task_id=task_id)

    return get_action

def load_pytorch_policy(fpath, itr, deterministic=False, return_model=False, task_id=0):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            # x = torch.as_tensor(x, dtype=torch.float32)
            x = {x_k: torch.as_tensor(x_v, dtype=torch.float32) for x_k, x_v in x.items()}
            action = model.act(x, task_id=task_id)
        return action
    if return_model:
        return model
    return get_action

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=100)
    parser.add_argument('--episodes', '-n', type=int, default=5)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--robot', type=int, default=0)
    parser.add_argument('--obstacle', type=int, default=0)
    parser.add_argument('--object', type=int, default=0)
    parser.add_argument('--task-id', type=int, default=0)
    args = parser.parse_args()
    get_action = load_policy(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic, task_id=args.task_id)

    env = create_env(args)


    run_policy(env, get_action, args.len, args.episodes, not(args.norender))