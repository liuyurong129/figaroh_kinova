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
import xml.etree.ElementTree as ET
from yaml.loader import SafeLoader
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
from figaroh.tools.robot import Robot
from figaroh.tools.regressor import build_regressor_basic, get_index_eliminate, build_regressor_reduced
from figaroh.tools.qrdecomposition import get_baseParams
from figaroh.identification.identification_tools import get_param_from_yaml, calculate_first_second_order_differentiation, base_param_from_standard, low_pass_filter_data

# Load robot model and create a dictionary containing reserved constants
ros_package_path = os.getenv('ROS_PACKAGE_PATH')
package_dirs = ros_package_path.split(':')
urdf_file = 'examples/airbot_play/data/urdf/airbot_play_v3_0_gripper.urdf'

robot = Robot(
    urdf_file,
    package_dirs = package_dirs
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data

with open('examples/airbot_play/config/airbot_play_config.yaml', 'r') as f:
    config = yaml.load(f, Loader=SafeLoader)
    pprint.pprint(config)
    
identif_data = config['identification']
params_settings = get_param_from_yaml(robot, identif_data)

# Count the number of samples in the CSV file
with open('examples/airbot_play/data/identification/chirp_motion/identification_q_real.csv', 'r') as f:
    row_count = sum(1 for _ in f)  
params_settings['nb_samples'] = row_count - 1 # Subtract 1 to account for the header row
print(params_settings)

params_standard_u = robot.get_standard_parameters(params_settings)
print(params_standard_u)

# Print out the placement of each joint of the kinematic tree
print("\nJoint placements:")
for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}"
          .format( name, *oMi.translation.T.flat )))

# Generate a list containing the full set of standard parameters
params_standard = robot.get_standard_parameters(params_settings)

# Build the structural base identification model, i.e., the one that can be observed, using random samples.

q_rand = np.random.uniform(low=-6, high=6, size=(params_settings["nb_samples"], model.nq))
dq_rand = np.random.uniform(low=-6, high=6, size=(params_settings["nb_samples"], model.nv))
ddq_rand = np.random.uniform(low=-6, high=6, size=(params_settings["nb_samples"], model.nv))

W = build_regressor_basic(robot, q_rand, dq_rand, ddq_rand, params_settings)
print(f"W.shape = {W.shape}, rank(W) = {np.linalg.matrix_rank(W)}")

idx_e, params_r = get_index_eliminate(W, params_standard, tol_e=1e-6)

W_e = build_regressor_reduced(W, idx_e)
print(f"W_e.shape = {W_e.shape}, rank(W_e) = {np.linalg.matrix_rank(W_e)}")

_, params_base, idx_base = get_baseParams(W_e, params_r, params_standard)

W_base = W_e[:, idx_base]
print(f"W_base.shape =  {W_base.shape}, rank(W_base) = {np.linalg.matrix_rank(W_base)}")
print("When using random trajectories the cond num is", np.linalg.cond(W_base))


# Read data from CSV and store into q
q = np.zeros((params_settings["nb_samples"], model.nq))
with open('examples/airbot_play/data/identification/chirp_motion/identification_q_real.csv', 'r') as f:
    csvreader = csv.reader(f)
    ii = 0
    for row in csvreader:
        if ii == 0:
            pass
        else: 
            q[ii-1, :] = np.array([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
        ii += 1

q_nofilt = np.array(q)
nbutter = 4
nbord = 5 * nbutter
q_filt = np.zeros((len(q) - 2 * 5 * nbutter, model.nq))

# Apply low pass filter to the data
for ii in range(model.nq):
    q_filt[:, ii] = low_pass_filter_data(q_nofilt[:, ii], params_settings, nbutter)


# Calculate first and second order differentiations
q, dq, ddq = calculate_first_second_order_differentiation(model, q_filt, params_settings)

# Compute nominal joint torques using RNEA (Recursive Newton-Euler Algorithm)
# Read updated friction parameters from the updated URDF file
def read_friction_params_from_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    friction_params = {}
    
    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        dynamics_elem = joint.find("dynamics")
        if dynamics_elem is not None:
            damping = dynamics_elem.get("damping", "0.0")
            friction = dynamics_elem.get("friction", "0.0")
            friction_params[joint_name] = {"damping": float(damping), "friction": float(friction)}
    
    return friction_params

# Read friction parameters from the updated URDF file
friction_params = read_friction_params_from_urdf(urdf_file)
print("Updated friction parameters from URDF:")
for joint, params in friction_params.items():
    print(f"{joint}: damping = {params['damping']}, friction = {params['friction']}")

tau_nominal = np.zeros((len(q), model.nq))

for i in range(len(q)):
    # Compute torque using RNEA
    tau_nominal[i, :] = pin.rnea(model, data, q[i, :], dq[i, :], ddq[i, :])
    
    # Add friction torques for each joint
    for j in range(model.nq):
        joint_name = f"joint{j+1}"
        if joint_name in friction_params:
            fv = friction_params[joint_name]["damping"]
            fs = friction_params[joint_name]["friction"]
        else:
            fv = 0.0
            fs = 0.0
        
        tau_fv = fv * dq[i, j]               # Viscous friction contribution
        tau_fs = fs * np.sign(dq[i, j])        # Static friction contribution
        tau_nominal[i, j] += tau_fv + tau_fs

tau_nominal = tau_nominal.T.flatten()

# Plot the second-order acceleration data
for i in range(ddq.shape[1]): 
    plt.plot(ddq[:, i], label=f'Joint {i+1} Acceleration')

plt.legend(title="Filtered Joint Acceleration")
plt.xlabel('Samples')
plt.ylabel('Acceleration')
plt.show()

# Build the regressor and analyze its properties
W = build_regressor_basic(robot, q, dq, ddq, params_settings)
print(f"W.shape = {W.shape}, rank(W) = {np.linalg.matrix_rank(W)}")

W_e = build_regressor_reduced(W, idx_e)
print(f"W_e.shape = {W_e.shape}, rank(W_e) = {np.linalg.matrix_rank(W_e)}")

W_base = W_e[:, idx_base]
print(f"W_base.shape =  {W_base.shape}, rank(W_base) = {np.linalg.matrix_rank(W_base)}")

print("When using chirp trajectories the cond num is", np.linalg.cond(W_base))

# Display structural base parameters
print("The structural base parameters are: ")
for ii in range(len(params_base)):
    print(params_base[ii])

# Load noise-contaminated torque data from CSV
tau_noised = np.empty(len(q) * model.nq)

with open('examples/airbot_play/data/identification/chirp_motion/identification_tau_real.csv', 'r') as f:
    csvreader = csv.reader(f)
    next(csvreader)  # Skip header
    tau_list = []

    for row in csvreader:
        tau_temp = np.array([float(value) for value in row[:6]])
        tau_list.append(tau_temp)

    tau_list = tau_list[:-2]
    tau_nofilt = np.array(tau_list)
    tau_filt = np.zeros((len(tau_list) - 2 * 5 * nbutter, model.nq))

    # Apply low pass filter to torque data
    for ii in range(model.nq):
        tau_filt[:, ii] = low_pass_filter_data(tau_nofilt[:, ii], params_settings, nbutter)

    for i in range(tau_filt.shape[1]):  # Iterate over each joint
        plt.plot(tau_filt[:, i], label=f'Joint {i+1} Torque')

    plt.legend(title="Filtered Joint Torque")
    plt.xlabel('Samples')
    plt.ylabel('Torque(Nm)')
    plt.show()

    tau_measured = np.array(tau_filt).T.flatten()

# Calculate structural base parameters using the regressor
phi_base = np.matmul(np.linalg.pinv(W_base), tau_measured)
print(f"phi_base_identified = {phi_base}")

phi_base_real = base_param_from_standard(params_standard, params_base)
print(f"phi_base_nominal = {phi_base_real}")
# Compute identified torque from base parameters
tau_identif = W_base @ phi_base

# Plot measured, identified, and nominal torque data
plt.plot(tau_measured, label='Measured Torque Data')
plt.plot(tau_identif, label='Identified Torque Data')
plt.plot(tau_nominal, label='Nominal Torque Data')
plt.xlabel('Samples')
plt.ylabel('Torque(Nm)')
plt.legend()
plt.show()

# Calculate RMSE and NRMSE for the identified torque
rmse_identified = np.sqrt(np.mean((tau_measured - tau_identif) ** 2))
nrmse_identified = (rmse_identified / np.sqrt(np.mean(tau_measured ** 2))) * 100
print(f"Root Mean Square Error (RMSE) for identified torque: {rmse_identified:.4f}")
print(f"Normalized Root Mean Square Error (NRMSE) for identified torque: {nrmse_identified:.4f}%")


# Calculate RMSE and NRMSE for the nominal torque
rmse_nominal = np.sqrt(np.mean((tau_measured - tau_nominal) ** 2))
nrmse_nominal = (rmse_nominal / np.sqrt(np.mean(tau_measured ** 2))) * 100
print(f"Root Mean Square Error (RMSE) for nominal torque: {rmse_nominal:.4f}")
print(f"Normalized Root Mean Square Error (NRMSE) for nominal torque: {nrmse_nominal:.4f}%")
