import pinocchio as pin
import numpy as np
import sys
import os
from os.path import join
import time
from pinocchio.visualize import GepettoVisualizer

# URDF文件路径
urdf_filename = "GEN3-7DOF-NOVISION_FOR_URDF_ARM_V12.urdf"
urdf_model_path = "examples/kinova_gen3/data/urdf"
urdf_path = join(urdf_model_path, urdf_filename)

# 确保文件路径正确
if not os.path.exists(urdf_path):
    print(f"URDF file not found: {urdf_path}")
    sys.exit(0)

# 加载模型（包含自由漂移基座，因此实际配置长度为 freeflyer+7）
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, urdf_model_path, pin.JointModelFreeFlyer())
data = model.createData()

# CSV 文件路径
csv_path = "examples/kinova_gen3/data/identification/trapezoidal_motion_7dof/identification_q_simulation.csv"
if not os.path.exists(csv_path):
    print(f"CSV file not found: {csv_path}")
    sys.exit(0)

# 加载 CSV 数据，假设第一行是标题，并且列之间以逗号分隔
# 注意：如果 CSV 文件中有多余的逗号（如 q1,q2,q3,q4,q5,q6,q7 之间不要额外逗号），需要提前清理
poses = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
print("Poses shape:", poses.shape)

# 设置采样率，用于控制显示速度（例如 125Hz）
sampling_rate = 100

# 创建并初始化 GepettoVisualizer
viz = GepettoVisualizer(model, collision_model, visual_model)
try:
    viz.initViewer()  # 初始化视图器
except ImportError as err:
    print("Error while initializing the viewer. It seems you should install gepetto-viewer")
    print(err)
    sys.exit(0)

try:
    viz.loadViewerModel("airbot")  # 加载机器人模型（名称随你实际情况修改）
except AttributeError as err:
    print("Error while loading the viewer model. It seems you should start gepetto-viewer")
    print(err)
    sys.exit(0)

# 显示机器人初始配置（free-flyer配置由 pin.neutral() 给出，关节位置在索引 7 之后）
q0 = pin.neutral(model)
viz.display(q0)

# 如果需要，也可以添加每个 link 坐标系的显示（可选）
frame_node_names = {}
for frame in model.frames:
    if frame.type == pin.FrameType.BODY:
        node_name = 'world/' + frame.name + '_axis'
        viz.viewer.gui.addXYZaxis(node_name, [1.0, 0.0, 0.0, 1.0], 0.03, 0.1)
        frame_id = model.getFrameId(frame.name)
        frame_node_names[frame_id] = node_name

# 播放轨迹
# 注意：模型中 free-flyer 的 7 个参数通常位于配置向量前7个位置，机器人关节从索引 7 开始
n_joints = poses.shape[1]
for i in range(poses.shape[0]):
    q_partial = poses[i, :]  # 6个关节数据
    q_full = pin.neutral(model)  # 获取全配置（包含 free-flyer）
    # 将CSV中采样的关节角度赋值到 free-flyer之后的部分
    q_full[7:7+n_joints] = q_partial
    
    # 计算前向运动学并更新frame位姿
    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    
    # 更新机器人显示
    viz.display(q_full)
    
    # 更新每个 link 坐标系显示（可选）
    for frame_id, node_name in frame_node_names.items():
        T = data.oMf[frame_id]
        config = pin.SE3ToXYZQUATtuple(T)
        viz.viewer.gui.applyConfiguration(node_name, config)
    
    time.sleep(1/sampling_rate)

print(f"Final poses shape: {poses.shape}")
