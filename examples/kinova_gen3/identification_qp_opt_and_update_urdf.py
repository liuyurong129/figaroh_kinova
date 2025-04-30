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
import re
import csv
import yaml
import pprint
import xml.etree.ElementTree as ET
from yaml import SafeLoader
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
from figaroh.tools.robot import Robot
from figaroh.tools.regressor import build_regressor_basic, get_index_eliminate, build_regressor_reduced
from figaroh.tools.qrdecomposition import get_baseParams
from figaroh.identification.identification_tools import get_param_from_yaml,calculate_first_second_order_differentiation, base_param_from_standard,low_pass_filter_data,relative_stdev


package_dirs = os.path.dirname(os.path.abspath(__file__))
robot_urdf_path = os.path.join(package_dirs, 'data', 'urdf', 'robot.urdf')
robot_new_urdf_path = os.path.join(package_dirs, 'data', 'urdf', 'robot_updated.urdf')
robot_new_config_path = os.path.join(package_dirs, 'config', 'robot_updated_config.yaml')
config_path = os.path.join(package_dirs, 'config', 'kinova_gen3_7dof_config.yaml')
identification_q_path = os.path.join(package_dirs, 'data','identification','trapezoidal_motion_7dof','q_trapezoid_real_weighted.csv')
identification_tau_path = os.path.join(package_dirs, 'data','identification','trapezoidal_motion_7dof','tau_trapezoid_real_weighted.csv')
package_dirs = "/home/bastienm/workspace/src/catkin_data_ws/install/share"

# Load robot model and create a dictionary containing reserved constants
ros_package_path = os.getenv('ROS_PACKAGE_PATH')
package_dirs = ros_package_path.split(':')

robot = Robot(
    robot_urdf_path,
    package_dirs = package_dirs
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data

with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=SafeLoader)
    pprint.pprint(config)
    
identif_data = config['identification']
params_settings = get_param_from_yaml(robot, identif_data)

# Count the number of samples in the CSV file
with open(identification_q_path, 'r') as f:
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
with open(identification_q_path, 'r') as f:
    csvreader = csv.reader(f)
    ii = 0
    for row in csvreader:
        if ii == 0:
            pass
        else: 
            q[ii-1, :] = np.array([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])])
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
friction_params = read_friction_params_from_urdf(robot_urdf_path)
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

with open(identification_tau_path, 'r') as f:
    csvreader = csv.reader(f)
    next(csvreader)  # Skip header
    tau_list = []

    for row in csvreader:
        tau_temp = np.array([float(value) for value in row[:7]])
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

print("Size of q samples:", q.shape)
print("Size of tau samples:", tau_measured.shape)

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

stdev=relative_stdev(W_base,phi_base, tau_measured)
print("\nRelative standard deviation:")
for ii in range(len(stdev)):
    print(f"{params_base[ii]} = {stdev[ii]:.4f}")


#############################################################

# QP Optimization

keys_template = ['Ixx', 'Ixy', 'Ixz', 'Iyy', 'Iyz', 'Izz', 'mx', 'my', 'mz', 'm', 'fv', 'fs']

# Extract URDF parameters
phi_urdf = []
num_links = max(int(k[-1]) for k in params_standard.keys() if k[-1].isdigit())

for i in range(1, num_links + 1):
    for key in keys_template:
        key_name = f"{key}{i}"
        if key_name in params_standard:
            phi_urdf.append(params_standard[key_name])
phi_urdf = np.array(phi_urdf)

# Store parameter names
phi_vars = []
for i in range(1, num_links + 1):
    for key in keys_template:
        key_name = f"{key}{i}"
        if key_name in params_standard:
            phi_vars.append(key_name)

# Define dimensions
n_phi = 12 * model.nq  # Expected to be 72 (6 links * 12 parameters)
n_y = model.nq * params_settings["nb_samples"]

# Generate sample data
np.random.seed(42)
W_qp = np.random.randn(n_y, n_phi)
Y = np.random.randn(n_y, 1)

# Compute QP matrices
H = W_qp.T @ W_qp  # Quadratic term
c = -W_qp.T @ Y    # Linear term

# Objective function
def objective(Phi, alpha=1000):
    qp_term = (0.5 * Phi.T @ H @ Phi + c.T @ Phi) / (params_settings["nb_samples"] * model.nq)
    phi_diff_term = alpha * np.sum((phi_urdf - Phi)**2)
    return qp_term + phi_diff_term

# Constraint 1: Mass terms (every 12th element) must be positive
constr1 = {'type': 'ineq', 'fun': lambda Phi: Phi[9::12]}

# Parse an expression into a dictionary of variable coefficients
def parse_expression(expr_str):
    """
    Parses a string like "Izz1 + 1.0*Iyy2 + ..." into {variable: coefficient, ...}.
    """
    expr_clean = re.sub(r'(?<!e)(-)', r'+-', expr_str)
    terms = [t.strip() for t in expr_clean.split('+') if t.strip()]
    coeffs = {}
    for term in terms:
        m = re.match(r'([+-]?\s*\d*\.?\d+(?:e[+-]?\d+)?)\s*\*\s*([A-Za-z]+\d+)', term)
        if m:
            coeff, var = float(m.group(1).replace(" ", "")), m.group(2).strip()
        else:
            var, coeff = term.replace(" ", ""), 1.0
        coeffs[var] = coeff
    return coeffs

# Define equality constraint matrix
n_eq = len(params_base)
A_auto = np.zeros((n_eq, n_phi))
b_auto = phi_base.copy()

# Populate A_auto
for i, expr in enumerate(params_base):
    coeffs = parse_expression(expr)
    for var, coeff in coeffs.items():
        if var in phi_vars:
            A_auto[i, phi_vars.index(var)] = coeff
        else:
            print(f"Warning: {var} not found in phi_vars")

# Constraint 2: Linear equality constraints from params_base
def constraint_phi_base(Phi):
    return A_auto.dot(Phi) - b_auto

constr2 = {'type': 'eq', 'fun': constraint_phi_base}

def compute_joint_com_constraints(robot_urdf_path, margin=0.00):
    model = pin.buildModelFromUrdf(robot_urdf_path)
    data = model.createData()
    q0 = pin.neutral(model)

    # Compute center of mass per link in world frame
    pin.centerOfMass(model, data, q0)
    # com_per_link = data.com[1:]  # index 0 is for universe

    COM_constraints = {}

    for joint_id in range(1, model.njoints):  # skip universe (id=0)
        joint_name = model.names[joint_id]
        child_id = model.getJointId(joint_name)

        # Get the CoM of the child link of the joint in world frame
        com_pos = data.oMi[child_id].act(data.com[child_id])

        COM_constraints[joint_id - 1] = {
            'x': (float(com_pos[0] - margin), float(com_pos[0] + margin)),
            'y': (float(com_pos[1] - margin), float(com_pos[1] + margin)),
            'z': (float(com_pos[2] - margin), float(com_pos[2] + margin)),
        }

    return COM_constraints

# Center of Mass (COM) constraints for each joint
# COM_constraints = {
#     0: {'x': (-0.0363, 0.0363), 'y': (-0.0604, 0.0604), 'z': (-0.0363, 0.0363)},
#     1: {'x': (-0.0363, 0.3264), 'y': (-0.0363, 0.0363), 'z': (-0.0604, 0.0604)},
#     2: {'x': (-0.0363, 0.0363), 'y': (-0.0363, 0.29015), 'z': (-0.0518, 0.0518)},
#     3: {'x': (-0.0544, 0.0544), 'y': (-0.0363, 0.0363), 'z': (-0.0907, 0.0363)},
#     4: {'x': (-0.0544, 0.0544), 'y': (-0.0363, 0.0864), 'z': (-0.0544, 0.0363)},
#     5: {'x': (-0.0544, 0.0544), 'y': (-0.0544, 0.136), 'z': (-0.2514, 0.0)}
# }

COM_constraints = compute_joint_com_constraints(robot_urdf_path)
print(f"COM constraints: {COM_constraints}")

# Constraint 3: Enforce COM limits using a linearized approximation
def center_of_mass_constraint(Phi):
    """
    For each link, apply a first-order Taylor expansion to approximate the nonlinear COM equation:
        COM = mx / m
    Linearized form:
        f_lin = (mx0/m0) + (1/m0) * (mx - mx0) - (mx0/m0^2) * (m - m0)
    """
    constraints = []
    for i in range(model.nq):
        base = i * 12

        # Extract reference values (Taylor expansion point)
        mx0, my0, mz0, m0 = phi_urdf[base + 6 : base + 10]
        if m0 == 0:
            raise ValueError(f"Linearization point mass m0 for link {i} cannot be 0")

        # Current variables
        mx, my, mz, m = Phi[base + 6 : base + 10]

        # Compute linear approximation
        f_lin_x = (mx0 / m0) + (1.0 / m0) * (mx - mx0) - (mx0 / m0**2) * (m - m0)
        f_lin_y = (my0 / m0) + (1.0 / m0) * (my - my0) - (my0 / m0**2) * (m - m0)
        f_lin_z = (mz0 / m0) + (1.0 / m0) * (mz - mz0) - (mz0 / m0**2) * (m - m0)

        # Apply scaled COM limits
        for axis, f_lin in zip(['x', 'y', 'z'], [f_lin_x, f_lin_y, f_lin_z]):
            lower, upper = [3 * v for v in COM_constraints[i][axis]]
            constraints.append(f_lin - lower)   # f_lin ≥ lower
            constraints.append(upper - f_lin)   # f_lin ≤ upper

    return np.array(constraints)

constr3 = {'type': 'ineq', 'fun': center_of_mass_constraint}

def compute_total_mass(robot_urdf_path):
    # Load the model from the URDF
    model = pin.buildModelFromUrdf(robot_urdf_path)

    # Sum the mass of all links
    total_mass = sum(inertia.mass for inertia in model.inertias)

    return total_mass

mass = compute_total_mass(robot_urdf_path)
print(f"Total mass from URDF: {mass}")

# Constraint 4: Ensure total mass equals 3.42405
def total_mass_constraint(Phi):
    return np.sum(Phi[9::12]) - mass  # Sum of all mass terms

constr4 = {'type': 'eq', 'fun': total_mass_constraint}

# Combine all constraints
constraints = [constr1, constr2, constr3, constr4]

# Initial estimate (72-dimensional vector)
Phi0 = phi_urdf  

# Select a range of alpha values (from 0 to 2, with 50 points)
alphas = np.linspace(0, 2, num=50)

qp_terms = []  # Store QP terms
phi_diff_terms = []  # Store phi difference terms
phi_opts = []  # Store optimized Phi values
tradeoff_values = []  # Store trade-off values

# Iterate over different alpha values and optimize the objective function
for a in alphas:
    res = minimize(lambda Phi: objective(Phi, alpha=a), Phi0, method='SLSQP', constraints=constraints)
    phi_opt_a = res.x
    
    # Compute QP term and phi difference term
    qp_term_a = (0.5 * phi_opt_a.T @ H @ phi_opt_a + c.T @ phi_opt_a) / params_settings["nb_samples"] / model.nq
    phi_diff_term_a = np.sum((phi_urdf - phi_opt_a)**2)
    
    # Store results
    qp_terms.append(qp_term_a)
    phi_diff_terms.append(phi_diff_term_a)
    phi_opts.append(phi_opt_a)
    
    # Compute trade-off value
    tradeoff_value = float(qp_term_a) - a * float(phi_diff_term_a)
    tradeoff_values.append(tradeoff_value)

    print(f"alpha: {a}, qp_term: {float(qp_term_a):.3f}, phi_diff_term: {float(phi_diff_term_a):.3f}, tradeoff: {tradeoff_value:.3f}")

# Select the optimal alpha (minimizing the absolute trade-off value)
best_alpha_index = np.argmin(np.abs(tradeoff_values))
best_alpha = alphas[best_alpha_index]
best_qp_term = qp_terms[best_alpha_index]
best_phi_diff_term = phi_diff_terms[best_alpha_index]

print(f"Best alpha: {best_alpha}, Best QP Term: {best_qp_term}, Best Phi Diff Term: {best_phi_diff_term}")

# Plot the Pareto front (QP term vs. phi difference term)
plt.figure(figsize=(6, 6))
plt.plot(qp_terms, phi_diff_terms, marker='o', linestyle='-', label='Pareto Front')
plt.scatter(best_qp_term, best_phi_diff_term, color='red', marker='x', label=f'Best alpha: {best_alpha}', zorder=2)
plt.xlabel('QP Term')
plt.ylabel('Phi Difference Term')
plt.title('Pareto Front: Trade-off between QP Term and Phi Difference Term')
plt.legend()
plt.grid(True)
plt.show()

# Solve QP again using the optimal alpha
result = minimize(lambda Phi: objective(Phi, alpha=best_alpha), Phi0, method='SLSQP', constraints=constraints)

# Optimal Phi (reshaped as a column vector)
Phi_opt = result.x.reshape(-1, 1)

# Extract required columns from W matrix (columns 1-10, 12, 13)
num_columns = W.shape[1]
indices_to_extract = [i+j for i in range(0, num_columns, 14) for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
W_filtered = W[:, indices_to_extract]

# Compute tau_pq
tau_pq = W_filtered @ Phi_opt

# Plot tau results from different methods
plt.title('Torque Comparison')
plt.plot(tau_identif, label='Identified')
plt.plot(tau_pq, label='QP')
plt.xlabel('Samples')
plt.ylabel('Torque(Nm)')
plt.legend()
plt.show()


#############################################################

# Compare identified parameters with URDF parameters

phi_urdf = np.asarray(phi_urdf).flatten()
Phi_opt = np.asarray(Phi_opt).flatten()
# print(phi_urdf, Phi_opt)

# Selected parameter indices and names
param_indices = [9, 6, 7, 8, 0, 3, 5]  # Mass, COM (x, y, z), Inertia (Ixx, Iyy, Izz)
param_names = ['Mass', 'COM x', 'COM y', 'COM z', 'Ixx', 'Iyy', 'Izz']

# Create subplots
fig, axes = plt.subplots(7, 1, figsize=(8, 18), sharex=True)

x_labels = [f'Joint {i+1}' for i in range(model.nq)]
x = np.arange(model.nq)

# Plot each parameter comparison
for i, (idx, param_name) in enumerate(zip(param_indices, param_names)):
    urdf_values = [phi_urdf[j * 12 + idx] for j in range(model.nq)]
    opt_values = [Phi_opt[j * 12 + idx] for j in range(model.nq)]
    
    ax = axes[i]
    width = 0.3
    ax.bar(x - width / 2, urdf_values, width=width, label='URDF', color='blue')
    ax.bar(x + width / 2, opt_values, width=width, label='Optimized', color='orange')
    
    ax.set_ylabel(param_name)
    ax.legend()
    ax.grid(axis='y')

# Set x-axis labels only on the last subplot
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(x_labels)

# Set plot title
fig.suptitle("Comparison of CAD and Identified SIP")
plt.tight_layout()
plt.show()


#############################################################

# updated URDF file

def FromDynamicParameters(params):
    inertia_matrices = []
    
    # Each object has 12 dynamic parameters
    num_objects = len(params) // 12
    
    for i in range(num_objects):
        # Extract dynamic parameters for each object
        v = params[i*12:(i+1)*12]
        
        # Extract mass, center of mass (COM), and inertia components
        m = v[9]  # Mass
        mc = v[6:9]  # COM position (m*c_x, m*c_y, m*c_z)
        
        # Extract inertia components (Ixx, Ixy, Ixz, Iyy, Iyz, Izz)
        Ixx, Ixy, Ixz, Iyy, Iyz, Izz = v[:6]
        
        # Construct the inertia matrix I_C (relative to COM)
        I_C = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz]
        ])
        
        S_c = np.array([
            [0, -mc[2], mc[1]],
            [mc[2], 0, -mc[0]],
            [-mc[1], mc[0], 0]
        ])

        # Compute S(c)^T * S(c)
        S_outer = S_c.T @ S_c
        
        # Update the inertia matrix
        inertia_matrix = I_C - S_outer / m
        
        inertia_matrices.append(inertia_matrix)
    
    return inertia_matrices

# Compute inertia matrices using identified parameters
inertia_matrices = FromDynamicParameters(Phi_opt)

# Print updated inertia matrices
for i, inertia_matrix in enumerate(inertia_matrices):
    print(f"Inertia Matrix {i+1}:")
    print(inertia_matrix)
    print()

tree = ET.parse(robot_urdf_path)
root = tree.getroot()

# Each object has 12 dynamic parameters
num_objects = len(Phi_opt) // 12
print(f"Number of objects: {num_objects}")

 # link_names = ["base_link", "shoulder_link", "half_arm_1_link", "half_arm_2_link", "forearm_link", 
    #              "spherical_wrist_1_link", "spherical_wrist_2_link", "bracelet_link", "end_effector_link", 
    #              "tool_frame", "FT_sensor_mounting", "FT_sensor_imu", "FT_sensor_wrench", "FT_sensor_mech1", 
    #              "FT_sensor_mech2", "FT_sensor_mech3", "FT_adapter"]
link_names = ["shoulder_link", "half_arm_1_link", "half_arm_2_link", "forearm_link", 
                "spherical_wrist_1_link", "spherical_wrist_2_link", "bracelet_link"]

# Update links and joints in the URDF file
for i in range(num_objects):
    # Extract dynamic parameters for each object
    v = Phi_opt[i*12:(i+1)*12]
    
    # Extract mass, COM position, and friction parameters
    m = v[9]  # Mass
    mc = v[6:9]  # COM position (m*c_x, m*c_y, m*c_z)
    fv = v[10]  # Friction parameter (damping)
    fs = v[11]  # Friction parameter (static friction)

    print(f"Link {i+1}:")
    print(f"  Mass: {m}")
    print(f"  COM: {mc}")
    print(f"  Friction: {fv}, {fs}")
    
    # Compute COM position (c_x, c_y, c_z)
    origin_xyz = mc / m
    
    # Add your own link names here
   
    link_name = link_names[i]
    print(f"Link name: {link_name}")
    link = root.find(f".//link[@name='{link_name}']")
    if link is not None:
        # Update origin
        origin_elem = link.find("inertial/origin")
        if origin_elem is not None:
            origin_elem.set("xyz", " ".join(map(str, origin_xyz)))
        
        # Update mass
        mass_elem = link.find("inertial/mass")
        if mass_elem is not None:
            mass_elem.set("value", str(m))
        
        # Update inertia matrix
        inertia_elem = link.find("inertial/inertia")
        if inertia_elem is not None:
            inertia_matrix = inertia_matrices[i]
            inertia_elem.set("ixx", str(inertia_matrix[0, 0]))
            inertia_elem.set("ixy", str(inertia_matrix[0, 1]))
            inertia_elem.set("ixz", str(inertia_matrix[0, 2]))
            inertia_elem.set("iyy", str(inertia_matrix[1, 1]))
            inertia_elem.set("iyz", str(inertia_matrix[1, 2]))
            inertia_elem.set("izz", str(inertia_matrix[2, 2]))
    
    # Update joint properties
    joint_name = f"joint{i+1}"
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is not None:
        # Find or create the <dynamics> element
        dynamics_elem = joint.find("dynamics")
        if dynamics_elem is None:
            dynamics_elem = ET.SubElement(joint, "dynamics")
        
        # Set friction parameters
        dynamics_elem.set("damping", str(fv))
        dynamics_elem.set("friction", str(fs))

# Save the updated URDF file
tree.write(robot_new_urdf_path, encoding="utf-8", xml_declaration=True)
print(f"Updated URDF file saved to {robot_new_urdf_path}")


#############################################################

# Compare identified, QP, and updated URDF torques

updated_robot = Robot(
    robot_new_urdf_path,
    package_dirs=package_dirs
)
updated_model = updated_robot.model
updated_data = updated_robot.data

with open(robot_new_config_path, 'r') as f:
    updated_config = yaml.load(f, Loader=SafeLoader)
    
updated_identif_data = updated_config['identification']
updated_params_settings = get_param_from_yaml(updated_robot, updated_identif_data)

# Generate a list containing the full set of standard parameters
updated_params_standard = updated_robot.get_standard_parameters(updated_params_settings)
print(updated_params_standard)


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
updated_friction_params = read_friction_params_from_urdf(robot_new_urdf_path)
print("Updated friction parameters from URDF:")
for joint, params in updated_friction_params.items():
    print(f"{joint}: damping = {params['damping']}, friction = {params['friction']}")

updated_tau_nominal = np.zeros((len(q), updated_model.nq))

for i in range(len(q)):
    # Compute torque using RNEA
    updated_tau_nominal[i, :] = pin.rnea(updated_model, updated_data, q[i, :], dq[i, :], ddq[i, :])
    
    # Add friction torques for each joint
    for j in range(updated_model.nq):
        joint_name = f"joint{j+1}"
        if joint_name in updated_friction_params:
            fv = updated_friction_params[joint_name]["damping"]
            fs = updated_friction_params[joint_name]["friction"]
        else:
            fv = 0.0
            fs = 0.0
        
        tau_fv = fv * dq[i, j]               # Viscous friction contribution
        tau_fs = fs * np.sign(dq[i, j])        # Static friction contribution
        updated_tau_nominal[i, j] += tau_fv + tau_fs

updated_tau_nominal = updated_tau_nominal.T.flatten()

# Plot torque comparison
plt.plot(tau_identif, label='Identified Torque')
plt.plot(tau_pq, label='QP Estimated Torque')
plt.plot(updated_tau_nominal, label='Updated URDF RNEA Torque')
plt.xlabel('Samples')
plt.ylabel('Torque(Nm)')
plt.title("Comparison of Torques: Identified, QP, and Updated URDF")  
plt.legend()
plt.show()