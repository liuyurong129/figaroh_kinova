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

from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer
import pinocchio as pin
from sys import argv
import numpy as np



class Robot(RobotWrapper):
    def __init__(
        self,
        robot_urdf,
        package_dirs,
        isFext=False,
        freeflyer_ori =None,
    ):
        # super().__init__()

        # intrinsic dynamic parameter names
        self.params_name = (
            "Ixx",
            "Ixy",
            "Ixz",
            "Iyy",
            "Iyz",
            "Izz",
            "mx",
            "my",
            "mz",
            "m",
        )

        # defining conditions
        self.isFext = isFext

        # folder location
        self.robot_urdf = robot_urdf

        # initializing robot's models
        if not isFext:
            self.initFromURDF(robot_urdf, package_dirs=package_dirs)
        else:
            self.initFromURDF(robot_urdf, package_dirs=package_dirs,
                              root_joint=pin.JointModelFreeFlyer())
        
        if freeflyer_ori is not None and isFext == True : 
            self.model.jointPlacements[self.model.getJointId('root_joint')].rotation = freeflyer_ori
            ub = self.model.upperPositionLimit
            ub[:7] = 1
            self.model.upperPositionLimit = ub
            lb = self.model.lowerPositionLimit
            lb[:7] = -1
            self.model.lowerPositionLimit = lb
            self.data = self.model.createData()

        # self.geom_model = pin.buildGeomFromUrdf(
        #     self.model, robot_urdf, geom_type=pin.GeometryType.COLLISION,
        #     package_dirs = package_dirs
        # )

        ## \todo test that this is equivalent to reloading the model
        self.geom_model = self.collision_model


    def get_standard_parameters(self, param):
        """This function prints out the standard inertial parameters defined in urdf model.
        Output: params_std: a dictionary of parameter names and their values"""
    
        model=self.model
        phi = []
        params = []

        params_name = (
            "Ixx",
            "Ixy",
            "Ixz",
            "Iyy",
            "Iyz",
            "Izz",
            "mx",
            "my",
            "mz",
            "m",
        )

        # change order of values in phi['m', 'mx','my','mz','Ixx','Ixy','Iyy','Ixz', 'Iyz','Izz'] - from pinoccchio
        # corresponding to params_name ['Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m']
        
        for i in range(1,len(model.inertias)):
            P =  model.inertias[i].toDynamicParameters()
            P_mod = np.zeros(P.shape[0])
            P_mod[9] = P[0]  # m
            P_mod[8] = P[3]  # mz
            P_mod[7] = P[2]  # my
            P_mod[6] = P[1]  # mx
            P_mod[5] = P[9]  # Izz
            P_mod[4] = P[8]  # Iyz
            P_mod[3] = P[6]  # Iyy
            P_mod[2] = P[7]  # Ixz
            P_mod[1] = P[5]  # Ixy
            P_mod[0] = P[4]  # Ixx
            for j in params_name:
                params.append(j + str(i))
            for k in P_mod:
                phi.append(k)
            
            params.extend(["Ia" + str(i)])
            params.extend(["fv" + str(i), "fs" + str(i)])
            params.extend(["off" + str(i)])
            
            if param['has_actuator_inertia']:
                try:
                    phi.extend([param['Ia'][i-1]])
                except Exception as e:
                    print("Warning: ", 'has_actuator_inertia_%d' % i, e)
                    phi.extend([0])
            else:
                phi.extend([0])
            if param['has_friction']:
                try:
                    phi.extend([param['fv'][i-1], param['fs'][i-1]])
                except Exception as e:
                    print("Warning: ", 'has_friction_%d' % i, e)
                    phi.extend([0, 0])
            else:
                phi.extend([0, 0])
            if param['has_joint_offset']:
                try:
                    phi.extend([param['off'][i-1]])
                except Exception as e:
                    print("Warning: ", 'has_joint_offset_%d' % i, e)
                    phi.extend([0])
            else:
                phi.extend([0]) 

        params_std = dict(zip(params, phi))
        return params_std

    def display_q0(self):
        """If you want to visualize the robot in this example,
        you can choose which visualizer to employ
        by specifying an option from the command line:
        GepettoVisualizer: -g
        MeshcatVisualizer: -m"""
        VISUALIZER = None
        if len(argv) > 1:
            opt = argv[1]
            if opt == "-g":
                VISUALIZER = GepettoVisualizer
            elif opt == "-m":
                VISUALIZER = MeshcatVisualizer

        if VISUALIZER:
            self.setVisualizer(VISUALIZER())
            self.initViewer()
            self.loadViewerModel(self.robot_urdf)
            q = self.q0

            self.display(q)
