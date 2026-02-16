#!/usr/bin/env python3
"""
Pick up a cube simulation using Franka Emika Panda robot.

This script loads up the Panda arm in MuJoCo, spawns a cube at a random spot,
and then uses inverse kinematics to figure out how to move the arm to grab it.
Pretty straightforward pick-and-place demo.
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
from dm_control.mujoco import Physics

# The Panda has 7 joints in its arm - these are their names in the MJCF file
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

# This is the body we want to control - the gripper/hand at the end of the arm
END_EFFECTOR_BODY = "hand"


def randomize_cube_position(physics):
    """
    Put the cube somewhere random on the table so each run is a bit different.
    We keep it within reach of the robot obviously.
    """
    # Random x,y position - staying within the robot's comfortable workspace
    x = np.random.uniform(0.43, 0.47)
    y = np.random.uniform(-0.03, 0.03)
    z = 0.02  # sitting on the floor (cube is 4cm tall, so center is at 2cm)
    
    # Give it a random rotation around Z so it's not always axis-aligned
    yaw = np.random.uniform(0, 2 * np.pi)
    # Convert yaw angle to quaternion - only rotating around Z axis here
    quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]  # MuJoCo uses [w, x, y, z] format
    
    # Now we need to actually set the cube's position in MuJoCo
    # The cube has a freejoint so we can move it anywhere - need to find where
    # its data lives in the qpos array
    cube_joint_id = physics.model.joint("cube_joint").id
    qpos_addr = physics.model.jnt_qposadr[cube_joint_id]
    
    # First 3 values are position, next 4 are quaternion
    physics.data.qpos[qpos_addr:qpos_addr+3] = [x, y, z]
    physics.data.qpos[qpos_addr+3:qpos_addr+7] = quat
    
    # Run forward kinematics so MuJoCo updates everything
    physics.forward()
    
    return np.array([x, y, z])


def get_cube_position(physics):
    """Just grab the current cube position from the simulation."""
    cube_joint_id = physics.model.joint("cube_joint").id
    qpos_addr = physics.model.jnt_qposadr[cube_joint_id]
    return physics.data.qpos[qpos_addr:qpos_addr+3].copy()


def compute_ik(physics, target_pos):
    """
    Figure out what joint angles we need to get the gripper to a target position.
    
    This uses the classic iterative Jacobian method - basically we:
    1. See where the gripper currently is
    2. Calculate the error from where we want it
    3. Use the Jacobian to figure out how to adjust joints
    4. Repeat until we're close enough
    
    We also try to keep the gripper pointing straight down since that's
    the best orientation for grabbing stuff off a table.
    """
    # Figure out which DOFs correspond to our arm joints
    dof_ids = [physics.model.joint(name).dofadr[0] for name in ARM_JOINTS]
    body_id = physics.model.body(END_EFFECTOR_BODY).id
    
    # Start from current configuration
    qpos = physics.data.qpos.copy()
    
    # Tuning parameters - these work reasonably well for this robot
    step_size = 0.4      # how aggressive each step is
    tolerance = 0.005    # 5mm is close enough
    max_iterations = 300 # don't loop forever if we can't reach it
    
    # We want the gripper pointing down (negative Z direction)
    target_z = np.array([0, 0, -1])
    
    # Make a separate copy of the physics data to do our IK iterations
    # without messing up the actual simulation state
    model = physics.model.ptr
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    for _ in range(max_iterations):
        # Where are we now?
        current_pos = data.xpos[body_id].copy()
        pos_error = target_pos - current_pos
        
        # What direction is the gripper currently pointing?
        rot_mat = data.xmat[body_id].reshape(3, 3)
        current_z = rot_mat[:, 2]  # z-axis of gripper in world frame
        
        # Cross product tells us which way to rotate to align with target
        # The 0.5 scaling just makes orientation correction less aggressive than position
        rot_error = np.cross(current_z, target_z) * 0.5
        
        # Stack position and rotation errors together
        error = np.concatenate([pos_error, rot_error])
        
        # Good enough? Then we're done
        if np.linalg.norm(pos_error) < tolerance:
            break
        
        # Compute the Jacobian - this tells us how joint velocities affect
        # end effector velocity (both linear and angular)
        jacp = np.zeros((3, model.nv))  # position jacobian
        jacr = np.zeros((3, model.nv))  # rotation jacobian
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        
        # We only care about our 7 arm joints, not any others
        jac = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
        
        # Damped least squares - adds stability when near singularities
        # (when the arm is stretched out or folded weird and the math gets sketchy)
        damping = 0.01
        jac_T = jac.T
        jac_pinv = jac_T @ np.linalg.inv(jac @ jac_T + damping * np.eye(6))
        
        # This is how much to move each joint
        dq = jac_pinv @ error * step_size
        
        # Apply the joint changes
        for i, dof_id in enumerate(dof_ids):
            data.qpos[dof_id] += dq[i]
        
        # Update forward kinematics for next iteration
        mujoco.mj_forward(model, data)
    
    # Extract and return the joint angles we found
    return np.array([data.qpos[dof_id] for dof_id in dof_ids])


def get_current_joint_positions(physics):
    """Read out where all the arm joints currently are."""
    positions = []
    for name in ARM_JOINTS:
        joint_id = physics.model.joint(name).id
        qpos_addr = physics.model.jnt_qposadr[joint_id]
        positions.append(physics.data.qpos[qpos_addr])
    return np.array(positions)


def move_to_position(physics, target_joints, steps=250, gripper_open=True, viewer=None):
    """
    Smoothly move the arm from where it is to the target joint positions.
    
    We interpolate between current and target over several steps so it
    doesn't just teleport (that would look weird and also might cause
    physics issues with the cube).
    """
    current = get_current_joint_positions(physics)
    
    # 255 = fully open, 0 = fully closed (it's just how the actuator is set up)
    gripper_val = 255 if gripper_open else 0
    
    for step in range(steps):
        # Linear interpolation - t goes from 0 to 1
        t = (step + 1) / steps
        interpolated = current + t * (target_joints - current)
        
        # Send commands to the robot
        # First 7 actuators are the arm joints, 8th is the gripper
        ctrl = physics.data.ctrl.copy()
        ctrl[:7] = interpolated
        ctrl[7] = gripper_val
        physics.set_control(ctrl)
        
        physics.step()
        if viewer:
            viewer.sync()
    
    # Let things settle down after we reach the target
    for _ in range(50):
        physics.step()
        if viewer:
            viewer.sync()


def control_gripper(physics, close=True, steps=50, viewer=None):
    """
    Open or close the gripper fingers.
    Keeps the arm where it is while doing this.
    """
    target = 0 if close else 255
    
    ctrl = physics.data.ctrl.copy()
    ctrl[7] = target
    
    for _ in range(steps):
        physics.set_control(ctrl)
        physics.step()
        if viewer:
            viewer.sync()


def main():
    """
    Main function - sets up the scene and runs through the pick-and-place sequence.
    """
    # We need to be in the robot's directory so the XML includes work properly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    robot_dir = os.path.join(script_dir, "franka_emika_panda")
    
    original_dir = os.getcwd()
    os.chdir(robot_dir)
    
    # This XML defines our whole scene - robot, floor, lighting, and the cube
    # We include the Panda robot from its XML file and add our own stuff around it
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
        
        <!-- The red cube we're going to pick up - 4cm on each side -->
        <body name="cube" pos="0.5 0 0.02">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="0.02 0.02 0.02" rgba="1 0 0 1" mass="0.1" friction="1 0.5 0.5"/>
        </body>
      </worldbody>
    </mujoco>
    """
    
    # Load up the simulation
    physics = Physics.from_xml_string(scene_xml)
    os.chdir(original_dir)
    
    print("Simulation loaded successfully")
    
    # Open a window so we can watch what's happening
    viewer = mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr)
    
    # Start the robot in a nice home position with gripper open
    # These values put the arm in a reasonable starting configuration
    home_ctrl = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])
    physics.set_control(home_ctrl)
    for _ in range(100):
        physics.step()
        viewer.sync()
    print("Robot at home position, gripper open")
    
    # Put the cube somewhere random so it's not always in the same spot
    cube_pos = randomize_cube_position(physics)
    viewer.sync()
    print(f"Cube position: {cube_pos}")
    
    # Step 1: Move to a position above the cube (approach from above)
    approach_pos = cube_pos.copy()
    approach_pos[2] = 0.2  # 20cm above the ground
    print("Moving to approach position...")
    approach_joints = compute_ik(physics, approach_pos)
    move_to_position(physics, approach_joints, viewer=viewer)
    
    # Step 2: Lower down to grab the cube
    # The 0.11m height puts the fingertips right around the cube
    grasp_pos = cube_pos.copy()
    grasp_pos[2] = 0.11
    print("Lowering to grasp position...")
    grasp_joints = compute_ik(physics, grasp_pos)
    move_to_position(physics, grasp_joints, viewer=viewer)
    
    # Give the physics a moment to settle before we try to grab
    for _ in range(100):
        physics.step()
        viewer.sync()
    
    # Step 3: Close the gripper to grab the cube
    print("Closing gripper...")
    control_gripper(physics, close=True, steps=200, viewer=viewer)
    
    # Step 4: Lift it up!
    lift_pos = grasp_pos.copy()
    lift_pos[2] = 0.3  # raise to 30cm
    print("Lifting cube...")
    lift_joints = compute_ik(physics, lift_pos)
    move_to_position(physics, lift_joints, gripper_open=False, viewer=viewer)
    
    # Did we actually pick it up? Check if the cube moved up
    final_cube_pos = get_cube_position(physics)
    if final_cube_pos[2] > cube_pos[2] + 0.05:
        print("SUCCESS: Cube picked up!")
    else:
        print("Cube pickup attempt completed")
    
    print(f"Final cube position: {final_cube_pos}")
    
    # Keep the window open so you can look around
    print("Close the viewer window to exit...")
    while viewer.is_running():
        viewer.sync()


if __name__ == "__main__":
    main()
