#!/usr/bin/env python3
"""
Pick up a cube simulation using Franka Emika Panda robot.
Uses dm_control for MuJoCo interface and inverse kinematics.
"""

import os
import numpy as np
import mujoco
from dm_control import mujoco as dm_mujoco
from dm_control.mujoco import Physics

# Robot joint names (excluding gripper)
ARM_JOINTS = [
    "joint1", "joint2", "joint3", "joint4", 
    "joint5", "joint6", "joint7"
]

# Gripper joint names
GRIPPER_JOINTS = ["finger_joint1", "finger_joint2"]

# End effector body name (no site defined in this model)
END_EFFECTOR_BODY = "hand"


def randomize_cube_position(physics):
    """Randomize cube position and orientation on the table."""
    # Random position within reachable area
    x = np.random.uniform(0.4, 0.6)
    y = np.random.uniform(-0.2, 0.2)
    z = 0.025  # On the table surface
    
    # Random orientation (rotation around z-axis)
    angle = np.random.uniform(0, 2 * np.pi)
    quat = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
    
    # Set cube pose
    cube_joint_id = physics.model.joint("cube_joint").id
    qpos_addr = physics.model.jnt_qposadr[cube_joint_id]
    
    physics.data.qpos[qpos_addr:qpos_addr+3] = [x, y, z]
    physics.data.qpos[qpos_addr+3:qpos_addr+7] = quat
    
    physics.forward()
    
    return np.array([x, y, z])


def get_cube_position(physics):
    """Get current cube position from simulation."""
    cube_joint_id = physics.model.joint("cube_joint").id
    qpos_addr = physics.model.jnt_qposadr[cube_joint_id]
    return physics.data.qpos[qpos_addr:qpos_addr+3].copy()


def compute_ik(physics, target_pos, target_quat=None):
    """
    Compute inverse kinematics for the end effector to reach target position.
    Uses simple iterative IK approach.
    """
    # Get joint indices
    joint_ids = [physics.model.joint(name).id for name in ARM_JOINTS]
    dof_ids = [physics.model.joint(name).dofadr[0] for name in ARM_JOINTS]
    
    # Get end effector body id
    body_id = physics.model.body(END_EFFECTOR_BODY).id
    
    # Save original joint positions to restore later
    original_qpos = physics.data.qpos.copy()
    
    # IK parameters
    step_size = 0.5
    tolerance = 0.01
    max_iterations = 100
    
    for _ in range(max_iterations):
        # Get current end effector position
        current_pos = physics.data.xpos[body_id].copy()
        
        # Compute position error
        error = target_pos - current_pos
        
        if np.linalg.norm(error) < tolerance:
            break
        
        # Get Jacobian
        jacp = np.zeros((3, physics.model.nv))
        mujoco.mj_jacBody(physics.model.ptr, physics.data.ptr, jacp, None, body_id)
        
        # Extract only the arm joint columns
        jac = jacp[:, dof_ids]
        
        # Compute pseudo-inverse
        jac_pinv = np.linalg.pinv(jac)
        
        # Compute joint velocity
        dq = jac_pinv @ error * step_size
        
        # Update joint positions
        for i, joint_name in enumerate(ARM_JOINTS):
            joint_id = physics.model.joint(joint_name).id
            qpos_addr = physics.model.jnt_qposadr[joint_id]
            physics.data.qpos[qpos_addr] += dq[i]
        
        physics.forward()
    
    # Get computed joint positions
    joint_positions = []
    for joint_name in ARM_JOINTS:
        joint_id = physics.model.joint(joint_name).id
        qpos_addr = physics.model.jnt_qposadr[joint_id]
        joint_positions.append(physics.data.qpos[qpos_addr])
    
    # Restore original positions so move_to_target can interpolate properly
    physics.data.qpos[:] = original_qpos
    physics.forward()
    
    return np.array(joint_positions)


def set_joint_targets(physics, target_positions):
    """Set joint positions directly (simplified control)."""
    for i, joint_name in enumerate(ARM_JOINTS):
        joint_id = physics.model.joint(joint_name).id
        qpos_addr = physics.model.jnt_qposadr[joint_id]
        physics.data.qpos[qpos_addr] = target_positions[i]
    
    physics.forward()


def move_to_target(physics, target_positions, steps=100):
    """Move robot joints to target positions gradually."""
    # Get current positions
    current_positions = []
    for joint_name in ARM_JOINTS:
        joint_id = physics.model.joint(joint_name).id
        qpos_addr = physics.model.jnt_qposadr[joint_id]
        current_positions.append(physics.data.qpos[qpos_addr])
    current_positions = np.array(current_positions)
    
    # Interpolate to target
    for step in range(steps):
        t = (step + 1) / steps
        positions = current_positions + t * (target_positions - current_positions)
        set_joint_targets(physics, positions)
        
        # Step simulation
        physics.step()


def control_gripper(physics, close=True):
    """Open or close the gripper."""
    gripper_pos = 0.0 if close else 0.04  # 0 = closed, 0.04 = open
    
    for joint_name in GRIPPER_JOINTS:
        joint_id = physics.model.joint(joint_name).id
        qpos_addr = physics.model.jnt_qposadr[joint_id]
        physics.data.qpos[qpos_addr] = gripper_pos
    
    physics.forward()
    
    # Step simulation to let gripper move
    for _ in range(50):
        physics.step()


def main():
    """Main simulation loop."""
    # Load the scene - need to work from robot directory for asset paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    robot_dir = os.path.join(script_dir, "franka_emika_panda")
    
    # Change to robot directory for correct asset loading
    original_dir = os.getcwd()
    os.chdir(robot_dir)
    
    # Scene XML with cube added to the robot scene
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
          <geom name="cube_geom" type="box" size="0.025 0.025 0.025" rgba="1 0 0 1" mass="0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    
    physics = Physics.from_xml_string(scene_xml)
    
    # Return to original directory
    os.chdir(original_dir)
    
    print("Simulation loaded successfully")
    
    # Open gripper initially
    control_gripper(physics, close=False)
    print("Gripper opened")
    
    # Randomize cube position
    cube_pos = randomize_cube_position(physics)
    print(f"Cube spawned at position: {cube_pos}")
    
    # Target position slightly above the cube for grasping
    grasp_height = 0.1  # Height above cube to approach
    target_pos = cube_pos.copy()
    target_pos[2] += grasp_height
    
    # Compute IK for approach position
    print("Computing inverse kinematics for approach position...")
    approach_joints = compute_ik(physics, target_pos)
    print(f"Approach joint targets: {approach_joints}")
    
    # Move to approach position
    print("Moving to approach position...")
    move_to_target(physics, approach_joints)
    
    # Lower to grasp position
    target_pos[2] = cube_pos[2] + 0.02  # Just above cube
    print("Computing IK for grasp position...")
    grasp_joints = compute_ik(physics, target_pos)
    
    print("Lowering to grasp position...")
    move_to_target(physics, grasp_joints)
    
    # Close gripper to pick up cube
    print("Closing gripper...")
    control_gripper(physics, close=True)
    
    # Lift the cube
    target_pos[2] += 0.15
    print("Computing IK for lift position...")
    lift_joints = compute_ik(physics, target_pos)
    
    print("Lifting cube...")
    move_to_target(physics, lift_joints)
    
    # Check if cube was picked up
    final_cube_pos = get_cube_position(physics)
    if final_cube_pos[2] > cube_pos[2] + 0.05:
        print("Successfully picked up the cube!")
    else:
        print("Cube pickup attempt completed")
    
    print(f"Final cube position: {final_cube_pos}")
    print("Simulation complete")


if __name__ == "__main__":
    main()
