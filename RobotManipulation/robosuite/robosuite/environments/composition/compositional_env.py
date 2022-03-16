from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.environments.composition.lift_subtask import LiftSubtask
from robosuite.utils.observables import Observable, sensor
import robosuite.utils.transform_utils as T
from robosuite.utils.transform_utils import convert_quat

import numpy as np

class CompositionalEnv(SingleArmEnv):
    def __init__(
        self,
        robots,       # first compositional axis
        object_type, # second compositional axis
        obstacle,    # third compositional axis
        task,        # fourth compositional axis
        env_configuration="default",
        controller_configs=None,
        mount_types="default",
        gripper_types="RethinkGripper",
        initialization_noise=None,
        use_camera_obs=True,
        use_object_obs=True,
        use_goal_obs=True,
        use_obstacle_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        reward_shaping=True
    ):
        if gripper_types != "RethinkGripper":
            raise ValueError('We want to keep all robots using the same gripper because otherwise they have different observation spaces')
        # task includes arena, robot, and objects of interest
        elif task == 'Lift':
            self.task = LiftSubtask(self, object_type, obstacle, reward_shaping=reward_shaping)
        else:
            raise ValueError('{} is not a valid task'.format(task))
        
        # Needed for Robosuite gym wrapper
        self.reward_scale = self.task.reward_scale
        self.use_object_obs = use_object_obs
        self.use_goal_obs = use_goal_obs
        self.use_obstacle_obs = use_obstacle_obs

        super().__init__(
            robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=mount_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        return self.task.reward(action)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()  

        self.task._load_model()
        self.model = self.task.model

    def _setup_references(self):
        super()._setup_references()
        self.task._setup_references()

    def _setup_observables(self):
        observables = super()._setup_observables()
        # return self.task._setup_observables(observables)
        pf = self.robots[0].robot_model.naming_prefix
        
        # for conversion to relative gripper frame
        @sensor(modality="ref")
        def world_pose_in_gripper(obs_cache):
            return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if\
                f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)
        sensors = [world_pose_in_gripper]
        names = ["world_pose_in_gripper"]
        enableds = [True]
        actives = [False]
    
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            modality = "object"

            # object-related observables
            @sensor(modality=modality)
            def obj_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.task.obj_body_id])

            @sensor(modality=modality)
            def obj_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.task.obj_body_id]), to="xyzw")
            
            @sensor(modality=modality)
            def obj_to_eef_pos(obs_cache):
                # Immediately return default value if cache is empty
                if any([name not in obs_cache for name in
                        ["obj_pos", "obj_quat", "world_pose_in_gripper"]]):
                    return np.zeros(3)
                obj_pose = T.pose2mat((obs_cache["obj_pos"], obs_cache["obj_quat"]))
                rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                obs_cache[f"obj_to_{pf}eef_quat"] = rel_quat
                return rel_pos

            @sensor(modality=modality)
            def obj_to_eef_quat(obs_cache):
                return obs_cache[f"obj_to_{pf}eef_quat"] if \
                    f"obj_to_{pf}eef_quat" in obs_cache else np.zeros(4)

            sensors += [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
            names += ['obj_pos', 'obj_quat', 'obj_to_eef_pos', 'obj_to_eef_quat']
            
            enableds += [True] * 4
            actives += [True] * 4

        if self.use_goal_obs:
            modality = "goal"

            # goal-related observables
            @sensor(modality=modality)
            def goal_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.task.goal_body_id])

            @sensor(modality=modality)
            def goal_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.task.goal_body_id]), to="xyzw")
            
            @sensor(modality=modality)
            def goal_to_eef_pos(obs_cache):
                # Immediately return default value if cache is empty
                if any([name not in obs_cache for name in
                        ["goal_pos", "goal_quat", "world_pose_in_gripper"]]):
                    return np.zeros(3)
                goal_pose = T.pose2mat((obs_cache["goal_pos"], obs_cache["goal_quat"]))
                rel_pose = T.pose_in_A_to_pose_in_B(goal_pose, obs_cache["world_pose_in_gripper"])
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                obs_cache[f"goal_to_{pf}eef_quat"] = rel_quat
                return rel_pos

            @sensor(modality=modality)
            def goal_to_eef_quat(obs_cache):
                return obs_cache[f"goal_to_{pf}eef_quat"] if \
                    f"goal_to_{pf}eef_quat" in obs_cache else np.zeros(4)
            
            # in principle, like other things, we could add the quat as well
            @sensor(modality=modality)
            def obj_to_goal(obs_cache):
                return obs_cache["goal_pos"] - obs_cache["obj_pos"] if \
                    "obj_pos" in obs_cache and "goal_pos" in obs_cache else np.zeros(3)

            sensors += [goal_pos, goal_quat, goal_to_eef_pos, goal_to_eef_quat, obj_to_goal]
            names += ['goal_pos', 'goal_quat', 'goal_to_eef_pos', 'goal_to_eef_quat', 'obj_to_goal']
            
            enableds += [True] * 5
            actives += [True] * 5

        if self.use_obstacle_obs:
            modality = "obstacle"

            # goal-related observables
            @sensor(modality=modality)
            def obstacle_pos(obs_cache):
                # return np.array(self.sim.data.body_xpos[self.task.obstacle_body_id]) \
                #     if self.task.obstacle_type is not None else np.zeros(3)
                return np.array(self.sim.data.body_xpos[self.task.obstacle_body_id])

            @sensor(modality=modality)
            def obstacle_quat(obs_cache):
                # return convert_quat(np.array(self.sim.data.body_xquat[self.task.obstacle_body_id]), to="xyzw") \
                #     if self.task.obstacle_type is not None else np.zeros(4)
                return convert_quat(np.array(self.sim.data.body_xquat[self.task.obstacle_body_id]))
            
            @sensor(modality=modality)
            def obstacle_to_eef_pos(obs_cache):
                # Immediately return default value if cache is empty
                if any([name not in obs_cache for name in
                        ["obstacle_pos", "obstacle_quat", "world_pose_in_gripper"]]):
                    return np.zeros(3)
                obstacle_pose = T.pose2mat((obs_cache["obstacle_pos"], obs_cache["obstacle_quat"]))
                rel_pose = T.pose_in_A_to_pose_in_B(obstacle_pose, obs_cache["world_pose_in_gripper"])
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                obs_cache[f"obstacle_to_{pf}eef_quat"] = rel_quat
                return rel_pos

            @sensor(modality=modality)
            def obstacle_to_eef_quat(obs_cache):
                return obs_cache[f"obstacle_to_{pf}eef_quat"] if \
                    f"obstacle_to_{pf}eef_quat" in obs_cache else np.zeros(4)

            sensors += [obstacle_pos, obstacle_quat, obstacle_to_eef_pos, obstacle_to_eef_quat]
            names += ['obstacle_pos', 'obstacle_quat', 'obstacle_to_eef_pos', 'obstacle_to_eef_quat']

            enableds += [True] * 4
            actives += [True] * 4

        # Create observables
        for name, s, enabled, active in zip(names, sensors, enableds, actives):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
                enabled=enabled,
                active=active
            )
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        self.task._reset_internal()

    def _check_success(self):
        raise NotImplementedError('Thought this wasnt needed')

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        self.task.visualize(vis_settings=vis_settings)


