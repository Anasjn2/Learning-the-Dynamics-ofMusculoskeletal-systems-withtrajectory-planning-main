import numpy as np
import scipy.io as sio
import time
import matplotlib.pyplot as plt

from casadi import *
from PDP import PDP
from JinEnv import JinEnv
from arm_files.arm_flexor import Arm2DVecEnv, Arm2DEnv

# Load robot arm environment
robot_arm_env = JinEnv.RobotArm()
robot_arm_env.initDyn(g=9.81)

# Create PDP SysID object
dt = 0.01
robot_arm_id = PDP.SysID()
robot_arm_id.setAuxvarVariable(robot_arm_env.dyn_auxvar)
robot_arm_id.setStateVariable(robot_arm_env.X)
robot_arm_id.setControlVariable(robot_arm_env.U)
dyn = robot_arm_env.X + dt * robot_arm_env.f
robot_arm_id.setDyn(dyn)

# Generate experimental data
num_batches = 10
num_horizon_steps = 30003

batch_inputs = robot_arm_id.getRandomInputs(n_batch=num_batches, lb=[-5,-5], ub=[5,5], horizon=num_horizon_steps)
batch_states = []
batch_statep = []
acceleration = []
actuation = []

env = Arm2DEnv(visualize=False)
observation = env.reset()
for j in range(num_batches):
    states = np.zeros((num_horizon_steps + 1, 4))
    statesp = np.zeros((num_horizon_steps + 1, 4))
    acc = np.zeros((num_horizon_steps + 1, 2))
    act = np.zeros((num_horizon_steps, 2))
    for i in range(num_horizon_steps + 1):
        actions = env.action_space.sample()
        observation, reward, done, info = env.step(actions)
        states[i, :] = [-pi/2+observation['joint_pos'].get('r_shoulder')[0], observation['joint_pos'].get('elbow')[0],
                         observation['joint_vel'].get('r_shoulder')[0], observation['joint_vel'].get('elbow')[0]]
        statesp[i, :] = [observation['joint_vel'].get('r_shoulder')[0], observation['joint_vel'].get('elbow')[0],
                          observation['joint_acc'].get('r_shoulder')[0], observation['joint_acc'].get('elbow')[0]]
        acc[i, :] = [observation['joint_acc'].get('r_shoulder')[0], observation['joint_acc'].get('elbow')[0]]
        if i > 0:
             act[i - 1, :] = [observation['forces'].get('shoulder_flexion_actuator')[0], observation['forces'].get('elbow_flexion_actuator')[0]]
            
    batch_states += [states]
    batch_statep += [statesp]
    acceleration += [acc]  
    actuation += [act]
    observation = env.reset(random_target=True, obs_as_dict=True)

# Save the experimental data to a MATLAB .mat file
robotarm_iodata = {'batch_inputs': actuation,
                   'batch_states': batch_states,
                   'batch_statesp': batch_statep,
                   'acceleration': acceleration,
                   'actuation': actuation}
sio.savemat('data/robotarm_iodata.mat', {'robotarm_iodata': robotarm_iodata})        
