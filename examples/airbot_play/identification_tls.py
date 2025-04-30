# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import csv
import pprint
import yaml
from yaml.loader import SafeLoader
import numpy as np
import matplotlib.pyplot as plt
from figaroh.tools.robot import Robot
from figaroh.tools.regressor import build_regressor_basic, get_index_eliminate, build_regressor_reduced, build_total_regressor_current
from figaroh.tools.qrdecomposition import get_baseParams
from figaroh.identification.identification_tools import get_param_from_yaml, calculate_first_second_order_differentiation, low_pass_filter_data


# Load robot model and create a dictionary containing reserved constants
# ros_package_path = os.getenv('ROS_PACKAGE_PATH')
# package_dirs = ros_package_path.split(':')

# robot = Robot(
#     'examples/airbot_play/data/urdf/airbot_play_v3_0_gripper.urdf',
#     package_dirs=package_dirs
#     # isFext=True  # add free-flyer joint at base
# )

package_dirs = os.path.dirname(os.path.abspath(__file__))
robot_urdf_path = os.path.join(package_dirs, 'data', 'urdf', 'airbot_play_v3_0_gripper.urdf')
# robot_new_urdf_path = os.path.join(package_dirs, 'data', 'urdf', 'robot_updated.urdf')
# robot_new_config_path = os.path.join(package_dirs, 'config', 'robot_updated_config.yaml')
config_path = os.path.join(package_dirs, 'config', 'airbot_play_config.yaml')
identification_q_path = os.path.join(package_dirs, 'data','identification','chirp_motion','identification_q_simulation.csv')
identification_tau_path = os.path.join(package_dirs, 'data','identification','chirp_motion','identification_tau_simulation.csv')
identification_q_loaded_path = os.path.join(package_dirs, 'data','identification','chirp_motion','identification_q_simulation_loaded.csv')
identification_tau_loaded_path = os.path.join(package_dirs, 'data','identification','chirp_motion','identification_tau_simulation_loaded.csv')
package_dirs = "/home/bastienm/workspace/src/catkin_data_ws/install/share"

# Load robot model and create a dictionary containing reserved constants
robot = Robot(
    robot_urdf_path,
    package_dirs=package_dirs
    # isFext=True  # add free-flyer joint at base
)

model = robot.model
data = robot.data

# Load configuration file
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=SafeLoader)
    pprint.pprint(config)
    
identif_data = config['identification']
params_settings = get_param_from_yaml(robot, identif_data)

# Count the number of samples in the CSV file
with open(identification_q_path, 'r') as f:
    row_count = sum(1 for _ in f)  
params_settings['nb_samples'] = row_count - 1  # Subtract 1 to account for the header row
print(params_settings)

# Get the robot's standard parameters
params_standard_u = robot.get_standard_parameters(params_settings)
print(params_standard_u)

# Print the placement of each joint in the kinematic tree
print("\nJoint placements:")
for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}"
          .format(name, *oMi.translation.T.flat)))

# Generate a list containing the full set of standard parameters
params_standard = robot.get_standard_parameters(params_settings)

# Build the structural base identification model using random samples.
q_rand = np.random.uniform(low=-6, high=6, size=(params_settings["nb_samples"], model.nq))
dq_rand = np.random.uniform(low=-6, high=6, size=(params_settings["nb_samples"], model.nv))
ddq_rand = np.random.uniform(low=-6, high=6, size=(params_settings["nb_samples"], model.nv))

W = build_regressor_basic(robot, q_rand, dq_rand, ddq_rand, params_settings)
print(f"W.shape = {W.shape}, rank(W) = {np.linalg.matrix_rank(W)}")

# Perform elimination to reduce the number of parameters
idx_e, params_r = get_index_eliminate(W, params_standard, tol_e=1e-6)

W_e = build_regressor_reduced(W, idx_e)
print(f"W_e.shape = {W_e.shape}, rank(W_e) = {np.linalg.matrix_rank(W_e)}")

_, params_base, idx_base = get_baseParams(W_e, params_r, params_standard)

W_base = W_e[:, idx_base]
print(f"W_base.shape =  {W_base.shape}, rank(W_base) = {np.linalg.matrix_rank(W_base)}")
print("When using random trajectories, the condition number is", np.linalg.cond(W_base))


# Read data from CSV and store it into q
q = np.zeros((params_settings["nb_samples"], model.nq))
with open(identification_q_path, 'r') as f:
    csvreader = csv.reader(f)
    ii = 0
    for row in csvreader:
        if ii == 0:
            pass
        else: 
            q[ii - 1, :] = np.array([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
        ii += 1

q_with_load = np.zeros((params_settings["nb_samples"], model.nq))
with open(identification_q_loaded_path, 'r') as f:
    csvreader = csv.reader(f)
    ii = 0
    for row in csvreader:
        if ii == 0:
            pass
        else: 
            q_with_load[ii - 1, :] = np.array([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
        ii += 1

# Apply low-pass filtering to remove noise
nbutter = 4
nbord = 5 * nbutter

q_nofilt = np.array(q)
q_filt = np.zeros((len(q) - 2 * 5 * nbutter, model.nq))
for ii in range(model.nq):
    q_filt[:, ii] = low_pass_filter_data(q_nofilt[:, ii], params_settings, nbutter)

q_with_load_nofilt = np.array(q_with_load)
q_with_load_filt = np.zeros((len(q_with_load) - 2 * 5 * nbutter, model.nq))
for ii in range(model.nq):
    q_with_load_filt[:, ii] = low_pass_filter_data(q_with_load_nofilt[:, ii], params_settings, nbutter)


# Calculate first and second order derivatives (velocities and accelerations)
q, dq, ddq = calculate_first_second_order_differentiation(model, q_filt, params_settings)
q_with_load, dq_with_load, ddq_with_load = calculate_first_second_order_differentiation(model, q_with_load_filt, params_settings)

# Build regressor for original parameters
W = build_regressor_basic(robot, q, dq, ddq, params_settings)
W_e = build_regressor_reduced(W, idx_e)
W_base = W_e[:, idx_base]
print("When using chirp trajectories, the condition number of W_base (without load) is:", np.linalg.cond(W_base))

# Build regressor for parameters with load
W_with_load = build_regressor_basic(robot, q_with_load, dq_with_load, ddq_with_load, params_settings)
W_e_with_load = build_regressor_reduced(W_with_load, idx_e)
W_base_with_load = W_e_with_load[:, idx_base]
print("When using chirp trajectories, the condition number of W_base (with load) is", np.linalg.cond(W_base_with_load))

# Load real identification data from CSV
tau_noised = np.empty(len(q) * model.nq)  
with open(identification_tau_path, 'r') as f:
    csvreader = csv.reader(f)
    next(csvreader)  # Skip the header row
    tau_list = []

    for row in csvreader:
        tau_temp = np.array([float(value) for value in row[:6]])
        tau_list.append(tau_temp)

    tau_list = tau_list[:-2]
    tau_nofilt = np.array(tau_list)
    tau_filt = np.zeros((len(tau_list) - 2 * 5 * nbutter, model.nq))

    for ii in range(model.nq):
        tau_filt[:, ii] = low_pass_filter_data(tau_nofilt[:, ii], params_settings, nbutter)
        
    tau_measured = np.array(tau_filt).T.flatten()


# Load real identification data with load
tau_noised_with_load = np.empty(len(q_with_load) * model.nq) 
with open(identification_tau_loaded_path, 'r') as f:
    csvreader = csv.reader(f)
    next(csvreader)  
    tau_list = []

    for row in csvreader:
        tau_temp = np.array([float(value) for value in row[:6]])
        tau_list.append(tau_temp)

    tau_list = tau_list[:-2]
    tau_nofilt = np.array(tau_list)
    tau_filt = np.zeros((len(tau_list) - 2 * 5 * nbutter, model.nq))

    for ii in range(model.nq):
        tau_filt[:, ii] = low_pass_filter_data(tau_nofilt[:, ii], params_settings, nbutter)

    tau_measured_with_load = np.array(tau_filt).T.flatten()


# Define gain scale for torque data
gs_true = 1  # Torque scaling factor
vsa = (tau_measured / gs_true).reshape(-1, 1)  # Measured torque without load
vsb = (tau_measured_with_load / gs_true).reshape(-1, 1)  # Measured torque with load
params_settings["nb_samples"] -= 2 * 5 * nbutter  # Adjust sample count for filtering

# Build regressor using measurements and parameters
W_tot, V_norm, residue = build_total_regressor_current(W_base, W_base_with_load, W_with_load, vsa, vsb, params_standard, params_settings)
print(V_norm)  # Print normalized regressor values

# Extract base parameters and scaling factors
phi_base = V_norm[:48]  # Base parameters
gs = V_norm[48:54]  # Scaling factors
phi_unknon = V_norm[54:63]  # Unknown parameters
phi_known = V_norm[63:64]  # Known parameters

# Print scaling factors gs
print("\ng_s:")
labels_gs = ['g_s1', 'g_s2', 'g_s3', 'g_s4', 'g_s5', 'g_s6']
for lab, val in zip(labels_gs, gs):
    print(f"{lab} = {val:.4f}")

# Extract load parameters from W_with_load
W_l_temp = np.zeros((len(W_with_load), 14))
for k in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    W_l_temp[:, k] = W_with_load[:, (params_settings['which_body_loaded']) * 14 + k]

# Eliminate redundant parameters
idx_e_temp, params_r_temp = get_index_eliminate(W_l_temp, params_standard, 1e-6)
W_e_l = build_regressor_reduced(W_l_temp, idx_e_temp)

# Perform torque identification
tau_ident_unload = W_base @ phi_base  # Identified torque without load
tau_ident_load = W_base_with_load @ phi_base + W_e_l @ phi_unknon + params_settings['mass_load'] * W_with_load[:, (params_settings['which_body_loaded']) * 14 + 9].reshape(len(W_with_load), 1) @ phi_known  # Identified torque with load

# Plot the first figure (without load)
plt.plot(tau_measured, label='Measured without load')
plt.plot(tau_ident_unload, label='Identified without load')
plt.title('Torque Identification without Load')
plt.xlabel('Samples')
plt.ylabel('Torque(Nm)')
plt.legend()
plt.show()

# Plot the second figure (with load)
plt.plot(tau_measured_with_load, label='Measured with load')
plt.plot(tau_ident_load, label='Identified with load')
plt.title('Torque Identification with Load')
plt.xlabel('Samples')
plt.ylabel('Torque(Nm)')
plt.legend()
plt.show()