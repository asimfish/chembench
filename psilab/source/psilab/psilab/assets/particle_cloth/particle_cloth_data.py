# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import weakref

import omni.physics.tensors.impl.api as physx
# omni.physics.tensors.impl.api.ParticleClothView
import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import TimestampedBuffer


class ParticleClothData:

    def __init__(self, root_physx_view: physx.ParticleClothView, device: str):
        """Initializes the rigid object data.

        Args:
            root_physx_view: The root rigid body view.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device
        # Set the root rigid body view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_physx_view: physx.ParticleClothView = weakref.proxy(root_physx_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # Obtain global physics sim view
        physics_sim_view = physx.create_simulation_view("torch")
        physics_sim_view.set_subspace_roots("/")
        gravity = physics_sim_view.get_gravity()
        # Convert to direction vector
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)

        # Initialize constants
        self.GRAVITY_VEC_W = gravity_dir.repeat(self._root_physx_view.count, 1)
        self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_physx_view.count, 1)

        # Initialize the lazy buffers.
        self._positions = TimestampedBuffer()
        # self._root_state_w = TimestampedBuffer()
        # self._root_link_state_w = TimestampedBuffer()
        # self._root_com_state_w = TimestampedBuffer()
        # self._body_acc_w = TimestampedBuffer()

    def update(self, dt: float):
        """Updates the data for the rigid object.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt


    default_positions: torch.Tensor = None

    # ##
    # # Names.
    # ##

    # body_names: list[str] = None
    # """Body names in the order parsed by the simulation view."""

    # ##
    # # Defaults.
    # ##

    # default_root_state: torch.Tensor = None
    # """Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame. Shape is (num_instances, 13).

    # The position and quaternion are of the rigid body's actor frame. Meanwhile, the linear and angular velocities are
    # of the center of mass frame.
    # """

    # default_mass: torch.Tensor = None
    # """Default mass read from the simulation. Shape is (num_instances, 1)."""

    # default_inertia: torch.Tensor = None
    # """Default inertia tensor read from the simulation. Shape is (num_instances, 9).

    # The inertia is the inertia tensor relative to the center of mass frame. The values are stored in
    # the order :math:`[I_{xx}, I_{xy}, I_{xz}, I_{yx}, I_{yy}, I_{yz}, I_{zx}, I_{zy}, I_{zz}]`.
    # """

    # ##
    # # Properties.
    # ##

    @property
    def positions(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and orientation are of the rigid body's actor frame. Meanwhile, the linear and angular
        velocities are of the rigid body's center of mass frame.
        """

        if self._positions.timestamp < self._sim_timestamp:
            # read data from simulation
            positions = self._root_physx_view.get_positions().reshape(self._root_physx_view.count, -1, 3).clone()
            # pose = self._root_physx_view.get_transforms().clone()
            # pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
            # velocity = self._root_physx_view.get_velocities()
            # # set the buffer data and timestamp
            # self._root_state_w.data = torch.cat((pose, velocity), dim=-1)
            self._positions= positions
            self._positions.timestamp = self._sim_timestamp
        return self._positions.data

    # @property
    # def root_link_state_w(self):
    #     """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

    #     The position, quaternion, and linear/angular velocity are of the rigid body root frame relative to the
    #     world.
    #     """
    #     if self._root_link_state_w.timestamp < self._sim_timestamp:
    #         # read data from simulation
    #         pose = self._root_physx_view.get_transforms().clone()
    #         pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
    #         velocity = self._root_physx_view.get_velocities().clone()

    #         # adjust linear velocity to link from center of mass
    #         velocity[:, :3] += torch.linalg.cross(
    #             velocity[:, 3:], math_utils.quat_rotate(pose[:, 3:7], -self.com_pos_b[:, 0, :]), dim=-1
    #         )
    #         # set the buffer data and timestamp
    #         self._root_link_state_w.data = torch.cat((pose, velocity), dim=-1)
    #         self._root_link_state_w.timestamp = self._sim_timestamp

    #     return self._root_link_state_w.data



    # self._physics_view.get_positions()