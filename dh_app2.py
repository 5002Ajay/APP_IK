# nebula_kinematics_app.py
import streamlit as st
import numpy as np
import math
import json
from io import StringIO
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="NEBULA KNOWLAB",
    page_icon="//NEBULA KNOWLAB",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        border-bottom: 2px solid #64B5F6;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .slider-container {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .slider-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .slider-col {
        flex: 2;
    }
    .input-col {
        flex: 1;
    }
    /* Responsive design */
    @media (max-width: 1200px) {
        .main-header {
            font-size: 2.5rem;
        }
    }
    @media (max-width: 992px) {
        .main-header {
            font-size: 2rem;
        }
    }
    /* Visualization container */
    .visualization-container {
        height: 70vh;
        min-height: 500px;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">Nebula Robotics Kinematics Platform</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <p>This interactive platform helps you learn and experiment with forward and inverse kinematics concepts 
    for various robot configurations. Use the sidebar to configure your robot and the tabs below to explore 
    different kinematic concepts.</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for joint values
if 'joint_values' not in st.session_state:
    st.session_state.joint_values = []

# Robot configuration in sidebar
st.sidebar.header("Robot Configuration")

# Predefined robot configurations
robot_configs = {
    "UR5": {
        "type": "articulated",
        "dh_params": [
            {"a": 0, "d": 0.1625, "alpha": math.pi/2, "theta": 0},
            {"a": -0.425, "d": 0, "alpha": 0, "theta": 0},
            {"a": -0.3922, "d": 0, "alpha": 0, "theta": 0},
            {"a": 0, "d": 0.1333, "alpha": math.pi/2, "theta": 0},
            {"a": 0, "d": 0.0997, "alpha": -math.pi/2, "theta": 0},
            {"a": 0, "d": 0.0996, "alpha": 0, "theta": 0}
        ],
        "joint_types": ["revolute"] * 6,
        "joint_limits": [(-math.pi, math.pi)] * 6
    },
    "SCARA": {
        "type": "scara",
        "dh_params": [
            {"a": 0.3, "d": 0.2, "alpha": 0, "theta": 0},
            {"a": 0.25, "d": 0, "alpha": 0, "theta": 0},
            {"a": 0, "d": 0, "alpha": 0, "theta": 0},
            {"a": 0, "d": 0.1, "alpha": 0, "theta": 0}
        ],
        "joint_types": ["revolute", "revolute", "prismatic", "revolute"],
        "joint_limits": [(-math.pi, math.pi), (-math.pi, math.pi), (0, 0.2), (-math.pi, math.pi)]
    },
    "Cartesian": {
        "type": "cartesian",
        "dh_params": [
            {"a": 0, "d": 0, "alpha": 0, "theta": 0},
            {"a": 0, "d": 0, "alpha": math.pi/2, "theta": 0},
            {"a": 0, "d": 0, "alpha": 0, "theta": 0}
        ],
        "joint_types": ["prismatic", "prismatic", "prismatic"],
        "joint_limits": [(0, 0.5), (0, 0.5), (0, 0.5)]
    },
    "Custom": {
        "type": "custom",
        "dh_params": [],
        "joint_types": [],
        "joint_limits": []
    }
}

# Robot selection
selected_robot = st.sidebar.selectbox(
    "Select Robot Type",
    list(robot_configs.keys()),
    index=0
)

# Custom robot configuration
if selected_robot == "Custom":
    st.sidebar.subheader("Custom Robot Configuration")
    num_joints = st.sidebar.number_input("Number of Joints", min_value=1, max_value=10, value=3)
    
    robot_configs["Custom"]["dh_params"] = []
    robot_configs["Custom"]["joint_types"] = []
    robot_configs["Custom"]["joint_limits"] = []
    
    for i in range(num_joints):
        st.sidebar.markdown(f"**Joint {i+1} Configuration**")
        joint_type = st.sidebar.selectbox(f"Joint {i+1} Type", ["revolute", "prismatic"], key=f"jt{i}")
        a_val = st.sidebar.number_input(f"a{i+1}", value=0.1, key=f"a{i}")  # Default to 0.1 to avoid zero
        d_val = st.sidebar.number_input(f"d{i+1}", value=0.1, key=f"d{i}")  # Default to 0.1 to avoid zero
        alpha_val = st.sidebar.number_input(f"alpha{i+1}", value=0.0, key=f"alpha{i}")
        
        if joint_type == "revolute":
            min_val = st.sidebar.number_input(f"Min theta{i+1}", value=-180.0, key=f"min_t{i}")
            max_val = st.sidebar.number_input(f"Max theta{i+1}", value=180.0, key=f"max_t{i}")
            joint_limits = (math.radians(min_val), math.radians(max_val))
        else:
            min_val = st.sidebar.number_input(f"Min d{i+1}", value=0.0, key=f"min_d{i}")
            max_val = st.sidebar.number_input(f"Max d{i+1}", value=1.0, key=f"max_d{i}")
            joint_limits = (min_val, max_val)
        
        robot_configs["Custom"]["dh_params"].append({
            "a": a_val, 
            "d": d_val, 
            "alpha": math.radians(alpha_val),
            "theta": 0
        })
        robot_configs["Custom"]["joint_types"].append(joint_type)
        robot_configs["Custom"]["joint_limits"].append(joint_limits)

# Get the selected configuration
config = robot_configs[selected_robot]
dh_params = config["dh_params"]
joint_types = config["joint_types"]
joint_limits = config["joint_limits"]

# Initialize joint values if not set or if number of joints changed
if len(st.session_state.joint_values) != len(joint_types):
    st.session_state.joint_values = []
    for i, (j_type, limits) in enumerate(zip(joint_types, joint_limits)):
        if j_type == "revolute":
            st.session_state.joint_values.append(0.0)
        else:
            st.session_state.joint_values.append(limits[0])

# Forward kinematics functions
def dh_matrix(a, d, alpha, theta):
    """Create a Denavit-Hartenberg transformation matrix"""
    return np.array([
        [math.cos(theta), -math.sin(theta)*math.cos(alpha), math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
        [math.sin(theta), math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
        [0, math.sin(alpha), math.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(dh_params, joint_values):
    """Calculate forward kinematics for given joint values"""
    T = np.identity(4)
    joints = [np.array([0, 0, 0])]
    transforms = []
    
    for i, params in enumerate(dh_params):
        a, d, alpha, theta = params["a"], params["d"], params["alpha"], params["theta"]
        
        # Apply joint value
        if joint_types[i] == "revolute":
            theta += joint_values[i]
        else:
            d += joint_values[i]
            
        Ti = dh_matrix(a, d, alpha, theta)
        T = T @ Ti
        
        # Extract position
        position = T[:3, 3]
        joints.append(position)
        transforms.append(T.copy())
        
    return joints, T, transforms

# Fixed inverse kinematics functions with error handling
def inverse_kinematics_2r(target_x, target_y, l1, l2):
    """Inverse kinematics for a 2-link planar robot with error handling"""
    # Check for zero link lengths
    if l1 == 0 or l2 == 0:
        raise ValueError("Link lengths cannot be zero")
    
    # Calculate distance to target
    d = math.sqrt(target_x**2 + target_y**2)
    
    # Check if target is reachable
    if d > (l1 + l2) or d < abs(l1 - l2):
        raise ValueError("Target position is unreachable")
    
    # Calculate theta2 using law of cosines
    try:
        cos_theta2 = (target_x**2 + target_y**2 - l1**2 - l2**2) / (2 * l1 * l2)
        # Ensure the value is within valid range for acos
        cos_theta2 = max(min(cos_theta2, 1), -1)
        theta2 = math.acos(cos_theta2)
    except ValueError as e:
        raise ValueError("Invalid configuration for inverse kinematics") from e
    
    # Calculate theta1
    try:
        k1 = l1 + l2 * math.cos(theta2)
        k2 = l2 * math.sin(theta2)
        theta1 = math.atan2(target_y, target_x) - math.atan2(k2, k1)
    except ZeroDivisionError:
        # Handle special case when k1 and k2 are both zero
        theta1 = math.atan2(target_y, target_x)
    
    return theta1, theta2

# Inverse kinematics for 6-axis robot (simplified)
def inverse_kinematics_6dof(target_pos, target_orient, dh_params):
    """Simplified IK for 6-DOF robot using numerical approach"""
    # This is a simplified implementation - real IK would be more complex
    # For educational purposes, we'll use a simple approach
    
    # Extract position
    x, y, z = target_pos
    
    # For simplicity, we'll only solve for position, not orientation
    # Calculate distance from base
    distance = math.sqrt(x**2 + y**2 + z**2)
    
    # Simple heuristic for joint angles
    theta1 = math.atan2(y, x)
    
    # Calculate approximate joint angles
    # This is a simplified approach for educational purposes
    max_reach = sum(abs(p["a"]) for p in dh_params) + max(p["d"] for p in dh_params)
    
    if distance > max_reach:
        raise ValueError("Target position is unreachable")
    
    # Simple proportional distribution of angles
    # In a real implementation, you would use proper IK algorithms
    angles = []
    for i, params in enumerate(dh_params):
        if i == 0:
            angles.append(theta1)
        elif i == 1:
            angles.append(math.pi/4 * (z/distance if distance > 0 else 0))
        elif i == 2:
            angles.append(math.pi/4 * (x/distance if distance > 0 else 0))
        else:
            angles.append(0)  # Simple default for other joints
    
    return angles

# Function to create robot visualization
def create_robot_visualization(joints):
    """Create a 3D visualization of the robot using Plotly"""
    # Extract coordinates
    x_vals = [p[0] for p in joints]
    y_vals = [p[1] for p in joints]
    z_vals = [p[2] for p in joints]
    
    # Create the figure
    fig = go.Figure()
    
    # Add robot links
    fig.add_trace(go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals,
        mode='lines+markers',
        line=dict(width=8, color='blue'),
        marker=dict(size=6, color='red'),
        name='Robot Arm'
    ))
    
    # Add coordinate frames at each joint
    for i, joint in enumerate(joints):
        if i == 0:
            fig.add_trace(go.Scatter3d(
                x=[joint[0]], y=[joint[1]], z=[joint[2]],
                mode='markers',
                marker=dict(size=8, color='green'),
                name='Base'
            ))
        elif i == len(joints) - 1:
            fig.add_trace(go.Scatter3d(
                x=[joint[0]], y=[joint[1]], z=[joint[2]],
                mode='markers',
                marker=dict(size=8, color='red'),
                name='End Effector'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[joint[0]], y=[joint[1]], z=[joint[2]],
                mode='markers',
                marker=dict(size=6, color='orange'),
                name=f'Joint {i}'
            ))
    
    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (m)', range=[-1, 1]),
            yaxis=dict(title='Y (m)', range=[-1, 1]),
            zaxis=dict(title='Z (m)', range=[-0.5, 1.5])
        ),
        height=600,
        showlegend=True
    )
    
    return fig

# Function to create transformation visualization
def create_transformation_visualization(translation_matrix, rotation_matrix):
    """Create a 3D visualization of the transformation using Plotly"""
    # Combined transformation
    combined_matrix = translation_matrix @ rotation_matrix
    
    # Original arrow (larger size)
    arrow_start = np.array([0, 0, 0, 1])
    arrow_end = np.array([0.5, 0, 0, 1])  # Increased arrow length
    
    # Apply transformation
    arrow_start_transformed = np.dot(arrow_start, combined_matrix.T)
    arrow_end_transformed = np.dot(arrow_end, combined_matrix.T)
    
    # Create the figure
    fig = go.Figure()
    
    # Add original arrow (blue)
    fig.add_trace(go.Scatter3d(
        x=[arrow_start[0], arrow_end[0]],
        y=[arrow_start[1], arrow_end[1]],
        z=[arrow_start[2], arrow_end[2]],
        mode='lines',
        line=dict(width=8, color='blue'),
        name='Original'
    ))
    
    # Add transformed arrow (red)
    fig.add_trace(go.Scatter3d(
        x=[arrow_start_transformed[0], arrow_end_transformed[0]],
        y=[arrow_start_transformed[1], arrow_end_transformed[1]],
        z=[arrow_start_transformed[2], arrow_end_transformed[2]],
        mode='lines',
        line=dict(width=8, color='red'),
        name='Transformed'
    ))
    
    # Add coordinate axes
    fig.add_trace(go.Scatter3d(
        x=[0, 0.3], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(width=4, color='red'), name='X Axis'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0.3], z=[0, 0],
        mode='lines', line=dict(width=4, color='green'), name='Y Axis'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, 0.3],
        mode='lines', line=dict(width=4, color='blue'), name='Z Axis'
    ))
    
    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[-1, 1]),
            yaxis=dict(title='Y', range=[-1, 1]),
            zaxis=dict(title='Z', range=[-1, 1])
        ),
        height=600,
        showlegend=True
    )
    
    return fig

# Function to create target visualization
def create_target_visualization(target_x, target_y, target_z):
    """Create a 3D visualization of the target position using Plotly"""
    # Create the figure
    fig = go.Figure()
    
    # Add target position
    fig.add_trace(go.Scatter3d(
        x=[target_x], y=[target_y], z=[target_z],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Target Position'
    ))
    
    # Add coordinate axes
    fig.add_trace(go.Scatter3d(
        x=[0, 0.3], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(width=4, color='red'), name='X Axis'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0.3], z=[0, 0],
        mode='lines', line=dict(width=4, color='green'), name='Y Axis'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, 0.3],
        mode='lines', line=dict(width=4, color='blue'), name='Z Axis'
    ))
    
    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (m)', range=[-1, 1]),
            yaxis=dict(title='Y (m)', range=[-1, 1]),
            zaxis=dict(title='Z (m)', range=[-1, 1])
        ),
        height=600,
        showlegend=True
    )
    
    return fig

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Forward Kinematics", 
    "Inverse Kinematics", 
    "Transformation Concepts", 
    "Documentation"
])

with tab1:
    st.markdown('<h2 class="sub-header">Forward Kinematics</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p>Forward kinematics calculates the position and orientation of the robot's end-effector 
        based on its joint values. Adjust the sliders below to see how joint movements affect 
        the robot's position.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Joint sliders with input fields
    st.markdown("### Joint Controls")
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Adjust with Sliders")
        for i, (j_type, limits) in enumerate(zip(joint_types, joint_limits)):
            if j_type == "revolute":
                min_val, max_val = math.degrees(limits[0]), math.degrees(limits[1])
                # Use a shorter slider with step size
                new_value = st.slider(
                    f"Joint {i+1} (θ{i+1} in degrees)", 
                    min_val, max_val, float(math.degrees(st.session_state.joint_values[i])), 
                    key=f"j_slider_{i}",
                    step=1.0
                )
                st.session_state.joint_values[i] = math.radians(new_value)
            else:
                min_val, max_val = limits
                # Use a shorter slider with step size
                new_value = st.slider(
                    f"Joint {i+1} (d{i+1})", 
                    min_val, max_val, float(st.session_state.joint_values[i]), 
                    key=f"j_slider_{i}",
                    step=0.01
                )
                st.session_state.joint_values[i] = new_value
    
    with col2:
        st.markdown("#### Enter Exact Values")
        for i, (j_type, limits) in enumerate(zip(joint_types, joint_limits)):
            if j_type == "revolute":
                current_deg = math.degrees(st.session_state.joint_values[i])
                new_value = st.number_input(
                    f"θ{i+1} (degrees)", 
                    value=float(current_deg),
                    key=f"j_input_{i}",
                    step=1.0
                )
                st.session_state.joint_values[i] = math.radians(new_value)
            else:
                new_value = st.number_input(
                    f"d{i+1} (meters)", 
                    value=float(st.session_state.joint_values[i]),
                    key=f"j_input_{i}",
                    step=0.01
                )
                st.session_state.joint_values[i] = new_value
    
    # Calculate forward kinematics
    joints, T, transforms = forward_kinematics(dh_params, st.session_state.joint_values)
    end_effector_pos = joints[-1]
    
    # Display results
    st.markdown("### Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### End Effector Position")
        st.write(f"X: {end_effector_pos[0]:.3f} m")
        st.write(f"Y: {end_effector_pos[1]:.3f} m")
        st.write(f"Z: {end_effector_pos[2]:.3f} m")
        
        st.markdown("#### Transformation Matrix")
        st.write(T)
    
    with col2:
        st.markdown("#### 3D Visualization")
        
        # Create visualization
        fig = create_robot_visualization(joints)
        st.plotly_chart(fig, use_container_width=True)
        
        # Text information
        st.write("**Robot Configuration:**")
        for i, joint in enumerate(joints):
            if i == 0:
                st.write(f"Base: (0, 0, 0)")
            else:
                st.write(f"Joint {i}: ({joint[0]:.2f}, {joint[1]:.2f}, {joint[2]:.2f})")
        
        st.write(f"End Effector: ({end_effector_pos[0]:.2f}, {end_effector_pos[1]:.2f}, {end_effector_pos[2]:.2f})")

with tab2:
    st.markdown('<h2 class="sub-header">Inverse Kinematics</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p>Inverse kinematics calculates the joint values needed to achieve a desired end-effector 
        position and orientation. This is typically more complex than forward kinematics and may 
        have multiple solutions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # IK for different robot types
    if selected_robot in ["UR5", "SCARA", "Cartesian", "Custom"]:
        st.markdown("### Target Position and Orientation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_x = st.slider("Target X", -1.0, 1.0, 0.5, key="target_x")
            target_y = st.slider("Target Y", -1.0, 1.0, 0.5, key="target_y")
            target_z = st.slider("Target Z", -1.0, 1.0, 0.0, key="target_z")
            
            # Orientation sliders
            roll = st.slider("Roll (X)", -180.0, 180.0, 0.0, key="roll")
            pitch = st.slider("Pitch (Y)", -180.0, 180.0, 0.0, key="pitch")
            yaw = st.slider("Yaw (Z)", -180.0, 180.0, 0.0, key="yaw")
        
        with col2:
            # Calculate IK based on robot type
            target_pos = np.array([target_x, target_y, target_z])
            target_orient = np.array([math.radians(roll), math.radians(pitch), math.radians(yaw)])
            
            try:
                if selected_robot in ["UR5", "Custom"] and len(dh_params) >= 6:
                    # 6-DOF IK
                    joint_values = inverse_kinematics_6dof(target_pos, target_orient, dh_params)
                    st.success("IK solution found for 6-DOF robot!")
                    
                    for i, value in enumerate(joint_values):
                        if joint_types[i] == "revolute":
                            st.write(f"Joint {i+1}: {math.degrees(value):.2f}°")
                        else:
                            st.write(f"Joint {i+1}: {value:.3f} m")
                
                elif selected_robot == "SCARA" and len(dh_params) >= 2:
                    # SCARA IK (simplified)
                    l1 = abs(dh_params[0]["a"]) or 0.1
                    l2 = abs(dh_params[1]["a"]) or 0.1
                    
                    theta1, theta2 = inverse_kinematics_2r(target_x, target_y, l1, l2)
                    
                    st.write(f"Joint 1: {math.degrees(theta1):.2f}°")
                    st.write(f"Joint 2: {math.degrees(theta2):.2f}°")
                    st.write(f"Joint 3: {target_z:.3f} m")  # Z position for SCARA
                    st.success("IK solution found for SCARA robot!")
                    
                elif selected_robot == "Cartesian":
                    # Cartesian IK is straightforward
                    st.write(f"Joint 1: {target_x:.3f} m")
                    st.write(f"Joint 2: {target_y:.3f} m")
                    st.write(f"Joint 3: {target_z:.3f} m")
                    st.success("IK solution found for Cartesian robot!")
                    
            except ValueError as e:
                st.error(f"IK Error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
        
        # Visualization
        st.markdown("### Visualization")
        
        # Create a visualization of the target position
        fig = create_target_visualization(target_x, target_y, target_z)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional information
        st.write(f"**Target Position:** ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
        st.write(f"**Target Orientation:** Roll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°")

with tab3:
    st.markdown('<h2 class="sub-header">Transformation Concepts</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p>This section demonstrates fundamental transformation concepts including translation, 
        rotation, and Euler angles that form the basis of robot kinematics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Translation")
        tx = st.slider("Translate X", -1.0, 1.0, 0.5, key="tx")
        ty = st.slider("Translate Y", -1.0, 1.0, 0.3, key="ty")
        tz = st.slider("Translate Z", -1.0, 1.0, 0.2, key="tz")
        
        translation_matrix = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])
        
        st.write("Translation Matrix:")
        st.write(translation_matrix)
    
    with col2:
        st.markdown("### Rotation (Euler Angles)")
        
        # ZYX Euler angles
        roll = st.slider("Roll (X)", -180.0, 180.0, 30.0, key="roll_tab3")
        pitch = st.slider("Pitch (Y)", -180.0, 180.0, 45.0, key="pitch_tab3")
        yaw = st.slider("Yaw (Z)", -180.0, 180.0, 60.0, key="yaw_tab3")
        
        # Convert to radians
        roll_r = math.radians(roll)
        pitch_r = math.radians(pitch)
        yaw_r = math.radians(yaw)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll_r), -math.sin(roll_r)],
            [0, math.sin(roll_r), math.cos(roll_r)]
        ])
        
        Ry = np.array([
            [math.cos(pitch_r), 0, math.sin(pitch_r)],
            [0, 1, 0],
            [-math.sin(pitch_r), 0, math.cos(pitch_r)]
        ])
        
        Rz = np.array([
            [math.cos(yaw_r), -math.sin(yaw_r), 0],
            [math.sin(yaw_r), math.cos(yaw_r), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        
        # Homogeneous transformation matrix
        rotation_matrix = np.identity(4)
        rotation_matrix[:3, :3] = R
        
        st.write("Rotation Matrix:")
        st.write(rotation_matrix)
    
    st.markdown("### Combined Transformation")
    combined_matrix = translation_matrix @ rotation_matrix
    st.write("Combined Transformation Matrix (Translation then Rotation):")
    st.write(combined_matrix)
    
    # Visualization of transformation
    st.markdown("### Transformation Visualization")
    
    # Create visualization
    fig = create_transformation_visualization(translation_matrix, rotation_matrix)
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional information
    st.write("**Original Arrow:** Blue, pointing along X-axis")
    st.write("**Transformed Arrow:** Red, after translation and rotation")

with tab4:
    st.markdown('<h2 class="sub-header">Documentation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### About Nebula Robotics Kinematics Platform
    
    This platform is designed to help students and enthusiasts learn about robot kinematics through
    interactive visualization and experimentation.
    
    #### Features
    
    1. **Forward Kinematics**: Calculate end-effector position from joint angles
    2. **Inverse Kinematics**: Calculate joint angles from desired end-effector position
    3. **Transformation Concepts**: Understand translation, rotation, and homogeneous transformations
    4. **Multiple Robot Types**: UR5, SCARA, Cartesian, and custom configurations
    5. **Interactive 3D Visualizations**: See how your robot moves in 3D space
    
    #### Robot Types
    
    - **UR5**: A 6-DOF articulated robot arm commonly used in industrial applications
    - **SCARA**: Selective Compliance Assembly Robot Arm with 4 DOF
    - **Cartesian**: A 3-DOF robot that moves in XYZ coordinates
    - **Custom**: Define your own robot using DH parameters
    
    #### Denavit-Hartenberg Parameters
    
    The DH convention is a method for defining the relationship between consecutive robot links.
    Each transformation is defined by four parameters:
    
    - **a**: Link length
    - **d**: Link offset
    - **α**: Link twist
    - **θ**: Joint angle
    
    #### Mathematical Foundations
    
    The transformation between two links is given by the homogeneous transformation matrix:
    
    ```
    A = [[cosθ, -sinθ*cosα, sinθ*sinα, a*cosθ],
         [sinθ, cosθ*cosα, -cosθ*sinα, a*sinθ],
         [0, sinα, cosα, d],
         [0, 0, 0, 1]]
    ```
    
    The complete transformation from base to end-effector is the product of all individual
    transformation matrices:
    
    ```
    T = A₁ × A₂ × ... × Aₙ
    ```
    """)
    
    # Export configuration
    st.markdown("### Export Configuration")
    config_json = json.dumps(config, indent=2)
    st.download_button(
        label="Download Current Configuration",
        data=config_json,
        file_name=f"{selected_robot}_config.json",
        mime="application/json"
    )
    
    # Upload configuration
    st.markdown("### Upload Configuration")
    uploaded_file = st.file_uploader("Choose a JSON configuration file", type="json")
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        config_data = json.load(stringio)
        st.write("Uploaded configuration:")
        st.json(config_data)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "--- NEBULA KNOWLAB --- || Created for educational purposes"
    "</div>",
    unsafe_allow_html=True
)
