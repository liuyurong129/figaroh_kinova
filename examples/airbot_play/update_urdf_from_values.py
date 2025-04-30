import xml.etree.ElementTree as ET

# URDF file path
urdf_path = "examples/airbot_play/data/urdf/airbot_play_v3_0_gripper.urdf"
updated_urdf_path = "examples/airbot_play/data/urdf/updated_robot.urdf"

# Mass and inertia parameters for links
updated_links = {
    "link1": {
        "origin": {"xyz": [-1.44137333e-02/4.57625082e-03, -9.66535482e-03/4.57625082e-03, 9.90085662e-03/4.57625082e-03]},
        "mass": 4.57625082e-03,
        "inertia": {"ixx": -0.05334407,"ixy":0.01350084,"ixz":-0.01937493,"iyy":-0.06290141,"iyz":-0.04026797,"izz":-0.08559817},
    },
    "link2": {
        "origin": {"xyz": [3.65634858e-01/2.66948606e-08, 5.55011934e-02/2.66948606e-08, 1.02939935e-02/2.66948606e-08]},
        "mass": 2.66948606e-08 ,
        "inertia": {"ixx": -119362.01838632,"ixy":760189.99119328,"ixz":140994.92028022,"iyy":-5012006.41791235,"iyz":21402.10178466,"izz":-5123429.28569052},
    },
    "link3": {
        "origin": {"xyz": [-2.48605594e-02/1.11692062e-01 , 1.90700234e-01/1.11692062e-01, 3.45562136e-02/1.11692062e-01]},
        "mass": 1.11692062e-01,
        "inertia": {"ixx": -0.32975759,"ixy":0.08080885,"ixz":0.01928608,"iyy":-0.04084674,"iyz":0.12197621,"izz":-0.411243},
    },
    "link4": {
        "origin": {"xyz": [6.64030053e-03/1.60989421e-01 , 1.53819161e-02/1.60989421e-01, 6.93613460e-02/1.60989421e-01]},
        "mass": 1.60989421e-01,
        "inertia": {"ixx": -0.01050437,"ixy":0.00428766,"ixz":0.00400393,"iyy":-0.02116603,"iyz":0.00328516,"izz":-0.00873932},
    },
    "link5": {
        "origin": {"xyz": [-1.86056320e-02/1.81802278e-01, 2.36424985e-02/1.81802278e-01, -6.90169050e-03/1.81802278e-01]},
        "mass": 1.81802278e-01,
        "inertia": {"ixx": 0.00039115,"ixy":-0.00900296,"ixz":0.00155151,"iyy":0.00696256,"iyz":0.00494687,"izz":-0.01900576},
    },
    "link6": {
        "origin": {"xyz": [1.11340239e-02/1.77171698e-01, 1.39075951e-02/1.77171698e-01, 1.80667933e-02/1.77171698e-01]},
        "mass": 1.77171698e-01,
        "inertia": {"ixx": -0.02160847,"ixy":-0.00266758,"ixz":-0.00743984,"iyy":-0.06176574,"iyz":0.01985647,"izz":-0.00248366},
    },
}

# Friction parameters for joints
updated_joints = {
    "joint1": {"fv": 1.75875295e-02 , "fs": 3.01544187e-01},
    "joint2": {"fv": -3.11416500e-01, "fs": 8.58795336e-01},
    "joint3": {"fv": 1.63482157e-02, "fs": 5.58490061e-01},
    "joint4": {"fv": 7.79252939e-02, "fs": 1.00262551e-01},
    "joint5": {"fv": 2.04964015e-02, "fs": 9.83198557e-02},
    "joint6": {"fv": 2.81589042e-02, "fs": 8.35191584e-02},
}

# Parse the URDF file
tree = ET.parse(urdf_path)
root = tree.getroot()

# Update mass and inertia for links
for link in root.findall("link"):
    link_name = link.get("name")
    if link_name in updated_links:
        params = updated_links[link_name]

        # Update origin
        origin_elem = link.find("inertial/origin")
        if origin_elem is not None:
            origin_elem.set("xyz", " ".join(map(str, params["origin"]["xyz"])))

        # Update mass
        mass_elem = link.find("inertial/mass")
        if mass_elem is not None:
            mass_elem.set("value", str(params["mass"]))

        # Update inertia
        inertia_elem = link.find("inertial/inertia")
        if inertia_elem is not None:
            for key, value in params["inertia"].items():
                inertia_elem.set(key, str(value))

# Update friction parameters for joints
for joint in root.findall("joint"):
    joint_name = joint.get("name")
    if joint_name in updated_joints:
        params = updated_joints[joint_name]

        # Find <dynamics> element, create if not present
        dynamics_elem = joint.find("dynamics")
        if dynamics_elem is None:
            dynamics_elem = ET.SubElement(joint, "dynamics")

        # Set friction parameters: fv (viscous friction), fs (static friction)
        dynamics_elem.set("damping", str(params["fv"]))
        dynamics_elem.set("friction", str(params["fs"]))

# Save the updated URDF file
tree.write(updated_urdf_path, encoding="utf-8", xml_declaration=True)
print(f"Updated URDF file saved to {updated_urdf_path}")
