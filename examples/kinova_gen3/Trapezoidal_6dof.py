# Copyright [2022-2023] [CNRS, Toward SAS]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import yaml
from yaml.loader import SafeLoader
from matplotlib import pyplot as plt
from figaroh.tools.robot import Robot
from figaroh.identification.identification_tools import get_param_from_yaml
from figaroh.tools.randomdata import get_torque_rand
import pinocchio as pin

class TrapezoidalTrajectory:
    def __init__(self, robot, active_joints: list, accel_ratio=0.25):
        """
        robot: Robot object from Airbot
        active_joints: List of active joint names, e.g., ["joint1", "joint2", ...]
        accel_ratio: The ratio of the acceleration/deceleration phase in the trapezoidal trajectory (default 0.25),
        i.e., δ₁/tf, and assume δ₁ = δ₃
        """
        self.robot = robot
        self.rmodel = self.robot.model
        # Note: In Airbot, internal joint indices are getJointId(j)-1
        self.active_joints = [(self.rmodel.getJointId(j) - 1) for j in active_joints]
        self.njoints = len(self.active_joints)
        self.accel_ratio = accel_ratio

        # Extract the position, velocity, torque and other constraints of the activated joint (only for activated joints)
        self.upper_q = self.rmodel.upperPositionLimit[self.active_joints]
        self.lower_q = self.rmodel.lowerPositionLimit[self.active_joints]
        self.upper_dq = self.rmodel.velocityLimit[self.active_joints]
        self.lower_dq = -self.rmodel.velocityLimit[self.active_joints]
        self.upper_effort = self.rmodel.effortLimit[self.active_joints]
        self.lower_effort = -self.rmodel.effortLimit[self.active_joints]

    def trapezoidal_segment(self, q0, qf, v_max, freq):
        """
        Generate a single trapezoidal trajectory (velocity rises uniformly from 0 to v_max, 
        holds v_max for a while, then uniformly decelerates back to 0).
        The total time T is dynamically calculated to ensure the speed is within the pos range.

        Parameters:
        q0: Initial position
        qf: Target position
        v_max: Maximum velocity
        freq: Sampling frequency (Hz)

        Returns:
        Time series, position, velocity, and acceleration arrays
        """
        D = abs(qf - q0)
        r = self.accel_ratio

        # Dynamically calculate T. The total time is determined by target position and speed limits
        T = D / v_max / (1 - r)  # Calculate the required time based on target position and peak speed

        # Time partitioning
        tf_accel = r * T
        tf_const = T - 2 * tf_accel

        # Calculate acceleration
        A = v_max / tf_accel

        # Time discretization
        N = int(T * freq) + 1
        t_vals = np.linspace(0, T, N)

        q_list = []
        v_list = []
        a_list = []

        # Determine motion direction
        sign = 1.0 if (qf - q0) >= 0 else -1.0

        for ti in t_vals:
            if ti <= tf_accel:  # Acceleration phase
                a_t = A * sign
                v_t = A * ti * sign
                q_t = q0 + 0.5 * A * ti**2 * sign
            elif ti <= tf_accel + tf_const:  # Constant velocity phase
                a_t = 0.0
                v_t = v_max * sign
                q_t = q0 + 0.5 * A * tf_accel**2 * sign + v_max * (ti - tf_accel) * sign
            else:  # Deceleration phase
                t_dec = ti - (tf_accel + tf_const)
                a_t = -A * sign
                v_t = v_max * sign - A * t_dec * sign
                q_t = (q0 + 0.5 * A * tf_accel**2 * sign +
                       v_max * tf_const * sign +
                       v_max * t_dec * sign - 0.5 * A * t_dec**2 * sign)

            q_list.append(q_t)
            v_list.append(v_t)
            a_list.append(a_t)

        return t_vals, np.array(q_list), np.array(v_list), np.array(a_list)


    def multi_trip_active_config(self, freq: int, num_round_trips: int, waypoints: np.ndarray, tf0_nominal: float):
        r = self.accel_ratio
        nj = self.njoints

        traj_joints = [[] for _ in range(nj)]
        for j in range(nj):
            q0_j = waypoints[0, j]
            qf_j = waypoints[1, j]
            D = abs(qf_j - q0_j)
            traj_j = []  # List to store trajectory for this joint (each segment is a dictionary with {'t':, 'q':, 'v':, 'a':, 'tf':})
            
            for i in range(num_round_trips):
                # Increase speed by 0.2 every two round trips, e.g., 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, ...
                speed_factor = 0.2 * ((i // 2) + 1)
                v_max = speed_factor * D / tf0_nominal
                print(f"Round trip {i+1}: v_max = {v_max:.4f}")  # Print maximum speed
                
                # Forward phase: q0 -> qf
                t_seg, q_seg, v_seg, a_seg = self.trapezoidal_segment(q0_j, qf_j, v_max, freq=freq)
                traj_j.append({'t': t_seg, 'q': q_seg, 'v': v_seg, 'a': a_seg, 'tf': t_seg[-1]})
                
                q_instant = q_seg[-1]
                
                # Return phase: qf -> q0
                t_seg_b, q_seg_b, v_seg_b, a_seg_b = self.trapezoidal_segment(q_instant, q0_j, v_max, freq=freq)
                traj_j.append({'t': t_seg_b, 'q': q_seg_b, 'v': v_seg_b, 'a': a_seg_b, 'tf': t_seg_b[-1]})
                
                q_instant = q_seg_b[-1]  # Update position
            
            traj_joints[j] = traj_j

        
        # When concatenating trajectories for all joints, since the durations of segments may differ, 
        # for simplicity, we assume all joints have similar motion distances D,
        # and use the segment duration of the first active joint as a reference.
        # We construct a global time axis (with segments joined end-to-end).
        t_global = []
        segment_boundaries = []
        t_curr = 0.0
        for seg in traj_joints[0]:
            t_seg = seg['t']
            # Concatenate the segment by shifting its time by t_curr
            t_global.extend((t_seg + t_curr).tolist())
            segment_boundaries.append(t_curr + t_seg[-1])
            t_curr += t_seg[-1]
        t_global = np.array(t_global)
        
        # For each joint, concatenate the trajectories of all segments, and concatenate the segments directly (no smooth transition between segments)
        p_all = []
        v_all = []
        a_all = []
        for j in range(nj):
            traj_j = traj_joints[j]
            q_concat = []
            v_concat = []
            a_concat = []
            t_accum = 0.0
            for seg in traj_j:
                t_seg = seg['t']
                q_concat.extend(seg['q'].tolist())
                v_concat.extend(seg['v'].tolist())
                a_concat.extend(seg['a'].tolist())
                t_accum += seg['tf']
            p_all.append(q_concat)
            v_all.append(v_concat)
            a_all.append(a_concat)

        # Transpose to get global trajectory matrix with shape = (N, njoints)
        p_act = np.array(p_all).T
        v_act = np.array(v_all).T
        a_act = np.array(a_all).T
        
        # Construct the global timestamp vector t_sample (N, 1)
        t_sample = t_global.reshape(-1, 1)
        return t_sample, p_act, v_act, a_act
    
    def check_cfg_constraints(self, q, v=None, tau=None, soft_lim=0):
        """
        Check if the generated trajectory violates position, velocity, and torque limits.
        If any violations are detected, raise an exception and print the position, velocity, and torque limits.
        """
        # Print position limits
        print("Position Limits:")
        for j in range(q.shape[1]):
            upper_limit = self.rmodel.upperPositionLimit[j]
            lower_limit = self.rmodel.lowerPositionLimit[j]
            print(f"Joint {j}: Lower Limit = {lower_limit}, Upper Limit = {upper_limit}")

        # Check position constraints
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                delta_lim = soft_lim * abs(self.rmodel.upperPositionLimit[j] - self.rmodel.lowerPositionLimit[j])
                upper_limit = self.rmodel.upperPositionLimit[j]
                lower_limit = self.rmodel.lowerPositionLimit[j]
                
                if q[i, j] > upper_limit - delta_lim:
                    raise ValueError(f"Position violation at joint {j}, value {q[i, j]} exceeds upper limit {upper_limit - delta_lim}")
                elif q[i, j] < lower_limit + delta_lim:
                    raise ValueError(f"Position violation at joint {j}, value {q[i, j]} below lower limit {lower_limit + delta_lim}")

        # Print velocity limits
        if v is not None:
            print("\nVelocity Limits:")
            for j in range(v.shape[1]):
                velocity_limit = self.rmodel.velocityLimit[j]
                print(f"Joint {j}: Velocity Limit = {velocity_limit}")
            
            # Check velocity constraints
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    velocity_limit = self.rmodel.velocityLimit[j]
                    if abs(v[i, j]) > (1 - soft_lim) * abs(velocity_limit):
                        raise ValueError(f"Velocity violation at joint {j}, value {v[i, j]} exceeds limit {velocity_limit}")

        # Print effort (torque) limits
        if tau is not None:
            print("\nEffort (Torque) Limits:")
            for j in range(tau.shape[1]):
                effort_limit = self.rmodel.effortLimit[j]
                print(f"Joint {j}: Effort Limit = {effort_limit}")
            
            # Check effort (torque) constraints
            for i in range(tau.shape[0]):
                for j in range(tau.shape[1]):
                    effort_limit = self.rmodel.effortLimit[j]
                    if abs(tau[i, j]) > (1 - soft_lim) * abs(effort_limit):
                        raise ValueError(f"Effort (torque) violation at joint {j}, value {tau[i, j]} exceeds limit {effort_limit}")

        # If no violations, return success
        print("\nSUCCEEDED to generate waypoints for a feasible trapezoidal trajectory")
        return False  # No violation


    def plot_spline(self, t, p, v, a):
        """
        Plot the trajectory for each active joint (position, velocity, and acceleration on the same plot).
        """
        q = p[:, self.active_joints]
        dq = v[:, self.active_joints]
        ddq = a[:, self.active_joints]
        
        for i in range(q.shape[1]):
            plt.figure(i)
            plt.plot(t[:, 0], q[:, i], color='r', label='pos')
            plt.plot(t[:, 0], dq[:, i], color='b', label='vel')
            plt.plot(t[:, 0], ddq[:, i], color='g', label='acc')
            plt.title(f'joint {i + 1}')
            plt.xlabel("Time (s)")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
        plt.show()


def main():
    # 1. Load the robot model and configuration
    ros_package_path = os.getenv('ROS_PACKAGE_PATH')
    package_dirs = ros_package_path.split(':')
    robot = Robot(
        'examples/kinova_gen3/data/urdf/GEN3-6DOF_NO-VISION_URDF_ARM_V01.urdf',
        package_dirs=package_dirs
    )
    # print("nq (size of q):", robot.model.nq)
    # print("nv (size of v):", robot.model.nv)
    # for i, joint in enumerate(robot.model.joints):
    #     print(f"Joint {i}: name = {robot.model.names[i]}, type = {type(joint).__name__}")

    with open('examples/kinova_gen3/config/kinova_gen3_6dof_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    identif_data = config['identification']
    params_settings = get_param_from_yaml(robot, identif_data)
    
    # 2. Define basic parameters: In this example, 6 active joints are used, and the trajectory is to go back and forth 5 times (i.e., forward and backward 5 times, totaling 10 segments)
    active_joints = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    freq = 100
    
    tf0_nominal = 2
    num_round_trips = 4

    # 3. Define the initial and target positions for the active joints (initial velocities are default to 0), convert the target angles to radians
    q0 = [-3, -2.0, -2.5, -3, -2, -3]
    qf = [ 3,  2.0,  2.5,  3,  2,  3]
    waypoints = np.vstack((q0, qf))   # Shape (2, 6)

    # 4. Instantiate the trapezoidal trajectory generator and generate the multi-segment back-and-forth trajectory
    TT_obj = TrapezoidalTrajectory(robot, active_joints, accel_ratio=0.25)
    t, p_full, v_full, a_full = TT_obj.multi_trip_active_config(freq, num_round_trips, waypoints, tf0_nominal)
    
    # 5. Use get_torque_rand to generate random torques (for constraint checking)
    tau = get_torque_rand(p_full.shape[0], robot, p_full, v_full, a_full, params_settings)
    tau = np.reshape(tau, (v_full.shape[1], v_full.shape[0])).transpose()
    TT_obj.check_cfg_constraints(p_full, v_full, tau, soft_lim=0.01)
    np.savetxt("examples/kinova_gen3/data/identification/trapezoidal_motion_6dof/identification_q_simulation.csv", p_full, delimiter=",", header="q1,q2,q3,q4,q5,q6", comments="")
    
    # 6. Plot the trajectory of the active joints (one plot per joint, showing displacement, velocity, and acceleration)
    TT_obj.plot_spline(t, p_full, v_full, a_full)
    
    print("Trajectory generation completed!")
    print("Trajectory time steps:", t.shape)

if __name__ == '__main__':
    main()
