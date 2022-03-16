from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *


if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = "CompositionalEnv" 

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    options["robots"] = choose_robots(exclude_bimanual=True)

    options["task"] = choose_task()
    options["object_type"] = choose_object().lower()
    options["obstacle"] = choose_obstacle()
    if options["obstacle"] == "None": options["obstacle"] = None

    # Choose controller
    controller_name = "OSC_POSITION"

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(10000):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        # for k, v in obs.items():
        #     print(k, v.shape)
        # exit()
        env.render()
