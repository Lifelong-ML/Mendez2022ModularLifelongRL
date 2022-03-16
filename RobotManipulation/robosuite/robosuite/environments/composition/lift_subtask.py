from collections import OrderedDict
import random
import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import (
    MilkObject,
    BreadObject,
    CerealObject,
    CanObject,
    WallObject,
    DoorFrameObject,
    BallObject,
)
from robosuite.models.objects import (
    MilkVisualObject,
    BreadVisualObject,
    CerealVisualObject,
    CanVisualObject
)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class LiftSubtask:
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        object_type (string): if provided, should be one of "milk", "bread", "cereal",
            or "can". Determines which type of object will be spawned on every
            environment reset. Only used if @single_object_mode is 2.
    
    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        parent_task,
        object_type,
        obstacle,
        table_full_size=(0.39, 0.49, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
    ):
        # parent task (e.g., CompositionalEnv)
        self.parent_task = parent_task

        # task settings
        self.object_type = object_type
        self.obstacle_type = obstacle
        self.object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
        self.object_id = self.object_to_id[object_type]
        self.obj_class_map = {
            "milk": (MilkObject, MilkVisualObject), 
            "bread": (BreadObject, BreadVisualObject), 
            "cereal": (CerealObject, CerealVisualObject),
            "can": (CanObject, CanVisualObject)}
        

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0.3, 0, 0.8))

        self.height_target = 0.1

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        self.reward_step_counter = 0

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.

        # use a shaping reward
        elif self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale

        self.reward_step_counter += 1
        return reward

    def staged_rewards(self):
        reach_mult = 0.1
        grasp_mult = 0.2
        lift_mult = 0.7

        # get reaching reward via minimum distance to a target object
        if self.obstacle_type is not None and 'wall' in self.obstacle_type.lower():
            # check if past wall
            wall_dist = self.parent_task._gripper_to_target(
                gripper=self.parent_task.robots[0].gripper,
                target=self.obstacle.root_body,
                target_type="body",
                return_distance=False
            )
            past_wall = wall_dist[0] < -0.05
            if past_wall:
                dist = self.parent_task._gripper_to_target(
                    gripper=self.parent_task.robots[0].gripper,
                    target=self.object.root_body,
                    target_type="body",
                    return_distance=True,
                )
                r_reach = reach_mult / 2 + (1 - np.tanh(5.0 * dist)) * (reach_mult / 2)
            else:
                gripper_pos = self.parent_task.sim.data.get_site_xpos(self.parent_task.robots[0].gripper.important_sites["grip_site"])
                gripper_pos_xz = np.array((gripper_pos[0], gripper_pos[2]))
                obstacle_pos = [-self.model.mujoco_arena.table_full_size[0] / 4, 0, 0] + self.table_offset
                obstacle_height = 0.1 + self.table_offset[2]
                target_xz = np.array((obstacle_pos[0] + 0.05, obstacle_height + 0.2))
                dist = np.linalg.norm(gripper_pos_xz - target_xz)
                r_reach = (1 - np.tanh(5.0 * dist)) * reach_mult / 2

        else:
            dist = self.parent_task._gripper_to_target(
                    gripper=self.parent_task.robots[0].gripper,
                    target=self.object.root_body,
                    target_type="body",
                    return_distance=True,
                )
            r_reach = (1 - np.tanh(5.0 * dist)) * reach_mult


        # grasping reward for touching any objects of interest
        r_grasp = int(self.parent_task._check_grasp(
            gripper=self.parent_task.robots[0].gripper,
            object_geoms=[g for g in self.object.contact_geoms])
        ) * grasp_mult

        # lifting reward for picking up an object
        r_lift = 0.
        if r_grasp > 0.:
            z_target = self.table_offset[2] + self.height_target
            object_z_loc = self.parent_task.sim.data.body_xpos[self.obj_body_id, 2]
            z_dist = np.maximum(z_target - object_z_loc, 0.)
            r_lift = grasp_mult + (1 - np.tanh(25.0 * z_dist)) * (
                    lift_mult - grasp_mult
            )

        return r_reach, r_grasp, r_lift

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """

        # Adjust base pose accordingly
        xpos = self.parent_task.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.parent_task.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.object = self.obj_class_map[self.object_type][0](self.object_type.title())
        self.visual_object = self.obj_class_map[self.object_type][1]('Visual' + self.object_type.title())
          

        mujoco_objects = [self.visual_object, self.object]
        if self.obstacle_type is None:
            self.obstacle =  BallObject('Obstacle'+str('None'), obj_type='visual', joints=None)
        elif 'wall' in self.obstacle_type.lower():
            self.obstacle = WallObject('Obstacle' + self.obstacle_type)
        elif 'door' in self.obstacle_type.lower():
            self.obstacle = DoorFrameObject('Obstacle' + self.obstacle_type)
        else:
            raise ValueError('Obstacle must be either a Wall or a DoorFrame')
        mujoco_objects.append(self.obstacle)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.parent_task.robots], 
            mujoco_objects=mujoco_objects,
        )

        self._get_placement_initializer()

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        
        # Add obstacle
        obstacle_pos = [-self.model.mujoco_arena.table_full_size[0] / 4, 0, 0] + self.table_offset
        rotation = 3.14 / 2
        rotation_axis = 'z'

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionObstacleSampler",
                mujoco_objects=self.obstacle,
                rotation=rotation,
                rotation_axis=rotation_axis,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=obstacle_pos
            )
        )
        obj_x_min = 0.
        obj_x_max = (self.model.mujoco_arena.table_full_size[0] / 2 - 0.15)
        obj_y_min = -(self.model.mujoco_arena.table_full_size[1] / 2 - 0.1)
        obj_y_max = -obj_y_min        
        goal_x_min = goal_x_max = goal_y_min = goal_y_max = 0



        fixed_object_rotation = 0.
        if self.object_type == 'cereal' and self.parent_task.robot_names[0] in ['Jaco', 'Sawyer', 'UR5e']:
            fixed_object_rotation = np.pi / 2
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.object,
                x_range=[obj_x_min, obj_x_max],
                y_range=[obj_y_min, obj_y_max],
                rotation=fixed_object_rotation,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        )

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"{self.visual_object.name}ObjectSampler",
                mujoco_objects=self.visual_object,
                x_range=[goal_x_min, goal_x_max],
                y_range=[goal_y_min, goal_y_max],
                # rotation=0,
                rotation=fixed_object_rotation,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=self.height_target
            )
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """

        # Additional object references from this env
        self.obj_body_id = self.parent_task.sim.model.body_name2id(self.object.root_body)
        self.visual_obj_body_id = self.parent_task.sim.model.body_name2id(self.visual_object.root_body)
        self.goal_body_id = self.visual_obj_body_id     # required for setting up observations in parent task
        
        self.obstacle_body_id = self.parent_task.sim.model.body_name2id(self.obstacle.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.parent_task.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                if "visual" in obj.name.lower():
                    self.parent_task.sim.model.body_pos[self.visual_obj_body_id] = obj_pos
                    self.parent_task.sim.model.body_quat[self.visual_obj_body_id] = obj_quat
                elif "obstacle" in obj.name.lower():
                    # Set the obstacle body locations
                    self.parent_task.sim.model.body_pos[self.obstacle_body_id] = obj_pos
                    self.parent_task.sim.model.body_quat[self.obstacle_body_id] = obj_quat
                else:
                    # Set the collision object joints
                    self.parent_task.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self.parent_task._visualize_gripper_to_target(gripper=self.parent_task.robots[0].gripper, target=self.object)

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        obj_height = self.parent_task.sim.data.body_xpos[self.obj_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        # cube is higher than the table top above a margin
        return obj_height > table_height + self.height_target
