## Pick up a Cube

In this task you will need to create a simulation environment and move a robot to pick up a cube.

# Sub-Tasks

1. Load a Franka Emika Panda robotic arm into Mujoco

2. Spawn a custom-sized cube into Mujoco (randomize position and orientation)

3. Generate the required joint states using inverse kinematics such that the robotics arm would be able to pick up the cube by closing its gripper

4. Move the Robot arm by setting the joint targets for the joint drives to the calculated joint states.

4. Once the target joint states have been reached close the Gripper and pick up the cube.


# Notes

- The files for loading the Franka Emika Panda robotic arm can be found here: https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda

- Use the package "dm_control" as the python interface to Mujoco, as it implements inverse kinematics for you and provides some quality of life functions.

- The cube can either be part of the MJCF file or be spawned into the environment dynamically using python

- Mujoco Documentation: https://mujoco.readthedocs.io/en/stable/overview.html

- Use Conda for environment managment

- All packages should be installable through pip

- For this task you can assume we know the cube position and orientation, so you can read those values directly from the simulation or store them for later use while spawning the cube.

- If anything is unclear send me an email and ask