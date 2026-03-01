# Pick Up a Cube — Franka Emika Panda + MuJoCo

A MuJoCo simulation that uses a **Franka Emika Panda** 7-DOF robotic arm to autonomously pick up a randomly-placed cube. The arm plans its motion with iterative Jacobian-based inverse kinematics and executes a full pick-and-place sequence rendered in a live 3-D viewer.

![Python 3.10](https://img.shields.io/badge/python-3.10-blue)
![MuJoCo](https://img.shields.io/badge/simulator-MuJoCo-orange)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **Franka Panda MJCF model** loaded from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)
- **Random cube spawning** — position and yaw are randomised each run so the task is never identical
- **Damped Least-Squares IK** — iterative Jacobian solver with orientation alignment (gripper kept pointing down)
- **Smooth joint-space interpolation** for natural-looking arm trajectories
- **Gripper control** — open / close the parallel-jaw gripper to grasp and lift the cube
- **Live 3-D viewer** via MuJoCo's passive viewer

## How It Works

1. The scene is built from an inline MJCF XML that includes the Panda model and adds a ground plane, lighting, and a 4 cm red cube with a free joint.
2. The cube is teleported to a random reachable pose on the ground.
3. The robot starts in a predefined home configuration with the gripper open.
4. **Approach** — IK computes the joint angles to place the hand 20 cm above the cube; the arm interpolates there.
5. **Lower** — IK computes a grasp height (≈ 11 cm above ground); the arm descends.
6. **Grasp** — The gripper closes around the cube.
7. **Lift** — The arm raises the cube to 30 cm and checks whether the pick was successful.

## Project Structure

```
pick-up-a-cube/
├── main.py                 # Entry point — simulation loop & pick-and-place logic
├── setup_robot.py          # Downloads the Panda MJCF model from MuJoCo Menagerie
├── environment.yml         # Conda environment specification
├── requirements.txt        # pip dependencies
├── LICENSE                 # MIT
├── franka_emika_panda/     # Panda robot model (MJCF XMLs + mesh assets)
│   ├── panda.xml
│   ├── hand.xml
│   ├── scene.xml
│   └── assets/             # STL/OBJ mesh files
└── archive/                # Original task description
    └── README.md
```

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 |
| Conda | any recent version |
| OS | Linux (tested), macOS, or Windows with a display server |
| GPU | Not required (MuJoCo is CPU-based) |

## Setup

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate pick-up-cube
```

Or install from pip directly:

```bash
pip install -r requirements.txt
```

### 2. Download the robot model

If the `franka_emika_panda/` directory is not already present:

```bash
python setup_robot.py
```

This fetches the Franka Panda model from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) repository and extracts it locally.

## Usage

```bash
conda activate pick-up-cube
python main.py
```

A MuJoCo viewer window will open showing the robot executing the following sequence:

1. Move to home position
2. Approach the cube from above
3. Lower to grasp height
4. Close the gripper
5. Lift the cube

Close the viewer window to exit the program.

## Key Implementation Details

### Inverse Kinematics

The IK solver in `compute_ik()` uses damped least-squares (Levenberg–Marquardt style) to iteratively converge on the target end-effector position:

$$\Delta q = J^T (J J^T + \lambda^2 I)^{-1} \, \Delta x \cdot \alpha$$

| Parameter | Value | Description |
|---|---|---|
| $\alpha$ | 0.4 | Step size per iteration |
| $\lambda^2$ | 0.01 | Damping factor for singularity robustness |
| Tolerance | 5 mm | Convergence threshold on position error |
| Max iterations | 300 | Upper bound on solver steps |

An orientation term steers the gripper's z-axis to point downward, weighted at 50 % relative to the position error.

### Motion Execution

Joint targets are linearly interpolated over 250 physics steps to produce smooth motion. After reaching the target, 50 additional settling steps are run.

### Gripper

The gripper actuator uses a single control value: **255** = fully open, **0** = fully closed.

## Dependencies

| Package | Purpose |
|---|---|
| [mujoco](https://pypi.org/project/mujoco/) | Physics engine & viewer |
| [dm_control](https://pypi.org/project/dm-control/) | Python bindings for MuJoCo (Physics wrapper) |
| [numpy](https://pypi.org/project/numpy/) | Numerical computation |

## License

This project is licensed under the [MIT License](LICENSE).
