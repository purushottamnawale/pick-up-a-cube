#!/usr/bin/env python3
"""
Pick up a cube simulation using Franka Emika Panda robot.
Uses dm_control for MuJoCo interface and inverse kinematics.
"""

import os
import numpy as np
import mujoco
from dm_control.mujoco import Physics

# Joint names
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

# End effector body name
END_EFFECTOR_BODY = "hand"


def randomize_cube_position(physics):
    """Randomize cube position on the table."""
    x = np.random.uniform(0.4, 0.5)
    y = np.random.uniform(-0.05, 0.05)
    z = 0.025  # On table surface
    
    # Set cube pose
    cube_joint_id = physics.model.joint("cube_joint").id
    qpos_addr = physics.model.jnt_qposadr[cube_joint_id]
    
    physics.data.qpos[qpos_addr:qpos_addr+3] = [x, y, z]
    physics.forward()
    
    return np.array([x, y, z])


def get_cube_position(physics):
    """Get current cube position."""
    cube_joint_id = physics.model.joint("cube_joint").id
    qpos_addr = physics.model.jnt_qposadr[cube_joint_id]
    return physics.data.qpos[qpos_addr:qpos_addr+3].copy()


def compute_ik(physics, target_pos):
    """
    Compute inverse kinematics using iterative Jacobian method.
    Returns joint positions that reach the target with gripper pointing down.
    """
    # Get joint DOF indices
    dof_ids = [physics.model.joint(name).dofadr[0] for name in ARM_JOINTS]
    body_id = physics.model.body(END_EFFECTOR_BODY).id
    
    # Work on a copy of qpos
    qpos = physics.data.qpos.copy()
    
    # IK parameters
    step_size = 0.4
    tolerance = 0.005
    max_iterations = 300
    
    # Target orientation: gripper pointing down (-Z world)
    # The hand's local Z axis should align with world -Z
    target_z = np.array([0, 0, -1])
    
    # Create temporary data for IK computation
    model = physics.model.ptr
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    for _ in range(max_iterations):
        current_pos = data.xpos[body_id].copy()
        pos_error = target_pos - current_pos
        
        # Get current orientation (rotation matrix)
        rot_mat = data.xmat[body_id].reshape(3, 3)
        current_z = rot_mat[:, 2]  # Local Z axis in world frame
        
        # Orientation error (cross product gives rotation axis * sin(angle))
        rot_error = np.cross(current_z, target_z) * 0.5
        
        # Combined error
        error = np.concatenate([pos_error, rot_error])
        
        if np.linalg.norm(pos_error) < tolerance:
            break
        
        # Compute Jacobians (position and orientation)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        
        # Stack position and rotation Jacobians
        jac = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
        
        # Damped pseudo-inverse for stability
        damping = 0.01
        jac_T = jac.T
        jac_pinv = jac_T @ np.linalg.inv(jac @ jac_T + damping * np.eye(6))
        dq = jac_pinv @ error * step_size
        
        # Update joint positions
        for i, dof_id in enumerate(dof_ids):
            data.qpos[dof_id] += dq[i]
        
        mujoco.mj_forward(model, data)
    
    # Return computed joint positions
    return np.array([data.qpos[dof_id] for dof_id in dof_ids])


def get_current_joint_positions(physics):
    """Get current arm joint positions."""
    positions = []
    for name in ARM_JOINTS:
        joint_id = physics.model.joint(name).id
        qpos_addr = physics.model.jnt_qposadr[joint_id]
        positions.append(physics.data.qpos[qpos_addr])
    return np.array(positions)


def move_to_position(physics, target_joints, steps=250, gripper_open=True):
    """Move robot to target joint positions using actuator control."""
    current = get_current_joint_positions(physics)
    gripper_val = 255 if gripper_open else 0
    
    for step in range(steps):
        t = (step + 1) / steps
        interpolated = current + t * (target_joints - current)
        
        # Set control for arm joints (actuators 0-6) and maintain gripper
        ctrl = physics.data.ctrl.copy()
        ctrl[:7] = interpolated
        ctrl[7] = gripper_val
        physics.set_control(ctrl)
        physics.step()
    
    # Settle at target
    for _ in range(50):
        physics.step()


def control_gripper(physics, close=True, steps=50):
    """Open or close the gripper."""
    # Actuator 8 (index 7): 0 = closed, 255 = open
    target = 0 if close else 255
    
    ctrl = physics.data.ctrl.copy()
    ctrl[7] = target
    physics.set_control(ctrl)
    
    for _ in range(steps):
        physics.step()


def main():
    """Main simulation loop."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    robot_dir = os.path.join(script_dir, "franka_emika_panda")
    
    original_dir = os.getcwd()
    os.chdir(robot_dir)
    
    # Scene with robot and cube
    scene_xml = """
    <mujoco model="panda_cube_scene">
      <include file="panda.xml"/>
      
      <statistic center="0.3 0 0.4" extent="1"/>
      
      <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-20"/>
      </visual>
      
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
          markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
      </asset>
      
      <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
        
        <!-- Cube to pick up -->
        <body name="cube" pos="0.5 0 0.025">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="0.025 0.025 0.025" rgba="1 0 0 1" mass="0.1" friction="1 0.5 0.5"/>
        </body>
      </worldbody>
    </mujoco>
    """
    
    physics = Physics.from_xml_string(scene_xml)
    os.chdir(original_dir)
    
    print("Simulation loaded successfully")
    
    # Initialize to home position and open gripper
    home_ctrl = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])
    physics.set_control(home_ctrl)
    for _ in range(100):
        physics.step()
    print("Robot at home position, gripper open")
    
    # Randomize cube position
    cube_pos = randomize_cube_position(physics)
    print(f"Cube position: {cube_pos}")
    
    # Move to approach position (above cube)
    approach_pos = cube_pos.copy()
    approach_pos[2] = 0.2
    print("Moving to approach position...")
    approach_joints = compute_ik(physics, approach_pos)
    move_to_position(physics, approach_joints)
    
    # Lower to grasp position (hand at ~0.115m puts fingertips properly around cube)
    grasp_pos = cube_pos.copy()
    grasp_pos[2] = 0.115
    print("Lowering to grasp position...")
    grasp_joints = compute_ik(physics, grasp_pos)
    move_to_position(physics, grasp_joints)
    
    # Close gripper
    print("Closing gripper...")
    control_gripper(physics, close=True, steps=200)
    
    # Lift cube
    lift_pos = grasp_pos.copy()
    lift_pos[2] = 0.3
    print("Lifting cube...")
    lift_joints = compute_ik(physics, lift_pos)
    move_to_position(physics, lift_joints, gripper_open=False)
    
    # Check result
    final_cube_pos = get_cube_position(physics)
    if final_cube_pos[2] > cube_pos[2] + 0.05:
        print("SUCCESS: Cube picked up!")
    else:
        print("Cube pickup attempt completed")
    
    print(f"Final cube position: {final_cube_pos}")


if __name__ == "__main__":
    main()
