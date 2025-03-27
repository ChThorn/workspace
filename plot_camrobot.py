#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import yaml
import os
import re

# Hardcoded file paths - update these to match your actual paths
WORKSPACE_FILE = "data/workspacenew.yaml"  # Changed from ../data to data
EXTRINSIC_FILE = "data/test_extrinsic.yaml"  # Changed from ../data to data

def parse_opencv_matrix(content):
    """Parse OpenCV matrix format from YAML content"""
    # Regular expression to find OpenCV matrices
    matrix_pattern = r'(?:bHc|camera_matrix|dist_coeffs):\s*!!opencv-matrix\s*rows:\s*(\d+)\s*cols:\s*(\d+)\s*dt:\s*\w+\s*data:\s*\[(.*?)\]'
    
    # Find all matrices in the content
    matches = re.findall(matrix_pattern, content, re.DOTALL)
    
    result = {}
    for match in matches:
        rows = int(match[0])
        cols = int(match[1])
        data_str = match[2].strip().replace('\n', ' ')
        data_list = [float(x) for x in data_str.split(',')]
        
        # Reshape into matrix
        matrix = np.array(data_list).reshape(rows, cols)
        
        # Find the name of this matrix
        pattern = r'(bHc|camera_matrix|dist_coeffs):\s*!!opencv-matrix'
        name_match = re.search(pattern, content)
        if name_match:
            name = name_match.group(1)
            result[name] = matrix
    
    return result

def load_yaml_file(file_path):
    """Load YAML file and return the data"""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            
            # Check if this is an OpenCV YAML file
            if '!!opencv-matrix' in content:
                # Parse OpenCV matrices if present
                opencv_matrices = parse_opencv_matrix(content)
                if 'bHc' in opencv_matrices:
                    return {'bHc': opencv_matrices['bHc']}
                
            # Otherwise parse as regular YAML
            if content.startswith('%YAML:1.0'):
                content = content.replace('%YAML:1.0', '')
            return yaml.safe_load(content)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def load_extrinsic_file(file_path):
    """Load extrinsic calibration file and return the transformation matrix"""
    data = load_yaml_file(file_path)
    if data and 'bHc' in data:
        bHc = np.array(data['bHc'], dtype=float)
        return bHc
    else:
        # Fallback to default matrix if file can't be loaded
        print("Warning: Could not load extrinsic calibration. Using default values.")
        # Identity rotation, translation [0.3, 0, 0.5] based on your output
        default_bHc = np.eye(4)
        default_bHc[0:3, 3] = [0.3, 0, 0.5]
        return default_bHc

def plot_coordinate_frame(ax, origin, R, scale=0.1, label_prefix=""):
    """Plot a coordinate frame with origin and rotation matrix R"""
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']
    
    # Plot coordinate axes
    for i in range(3):
        ax.quiver(origin[0], origin[1], origin[2],
                 R[0, i]*scale, R[1, i]*scale, R[2, i]*scale,
                 color=colors[i], label=f"{label_prefix}{labels[i]}")
        
        # Add text labels
        text_pos = origin + R[:3, i] * scale * 1.1
        ax.text(text_pos[0], text_pos[1], text_pos[2], f"{label_prefix}{labels[i]}")

def plot_workspace(ax, markers, color, label_prefix=""):
    """Plot markers and connect them to form the workspace"""
    # Extract x, y, z coordinates
    xs = [marker[0] for marker in markers]
    ys = [marker[1] for marker in markers]
    zs = [marker[2] for marker in markers]
    
    # Plot markers
    ax.scatter(xs, ys, zs, color=color, s=100, marker='o', label=f"{label_prefix}Markers")
    
    # Connect markers with lines to form the workspace bottom face
    for i in range(len(markers)):
        next_i = (i + 1) % len(markers)
        ax.plot([xs[i], xs[next_i]], [ys[i], ys[next_i]], [zs[i], zs[next_i]], color=color)
    
    # Add marker labels
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        ax.text(x, y, z, f'{label_prefix}M{i}', color=color)

def plot_workspace_volume(ax, markers, height=0.6, color='red', alpha=0.1, label="Workspace Volume"):
    """
    Modified version: Plot a 3D prism by computing the best-fit plane for the markers and extruding along its normal.
    This ensures that each extrusion (from a marker to the corresponding ceiling point) is done along the same direction,
    making the top face (ceiling) a parallel translation of the base.
    """
    markers = np.array(markers)
    # Compute the centroid of the markers
    centroid = np.mean(markers, axis=0)
    # Use SVD to compute the best-fit plane normal (the direction of smallest variance)
    U, S, Vt = np.linalg.svd(markers - centroid)
    normal = Vt[-1]  # plane normal
    
    # Project each marker onto the best-fit plane to get a consistent base
    bottom_corners = []
    for p in markers:
        d = np.dot(p - centroid, normal)
        p_proj = p - d * normal
        bottom_corners.append(p_proj)
    bottom_corners = np.array(bottom_corners)
    
    # Compute the top face by extruding the base along the computed normal
    top_corners = bottom_corners + height * normal

    # Define the faces of the prism
    faces = []
    faces.append(bottom_corners.tolist())  # Bottom face
    faces.append(top_corners.tolist())       # Top face
    num_pts = len(bottom_corners)
    for i in range(num_pts):
        j = (i + 1) % num_pts
        face = [bottom_corners[i].tolist(), bottom_corners[j].tolist(),
                top_corners[j].tolist(), top_corners[i].tolist()]
        faces.append(face)
    
    # Plot each face
    for face in faces:
        poly = Poly3DCollection([face], alpha=alpha, facecolor=color, edgecolor=color, linewidth=1)
        ax.add_collection3d(poly)
    
    # Draw extrusion (vertical) lines for clarity
    for i in range(num_pts):
        ax.plot([bottom_corners[i][0], top_corners[i][0]],
                [bottom_corners[i][1], top_corners[i][1]],
                [bottom_corners[i][2], top_corners[i][2]], color=color, alpha=0.7)
    
    # Label the floor and ceiling (using the average positions)
    ax.text(np.mean(bottom_corners[:,0]), np.mean(bottom_corners[:,1]), np.mean(bottom_corners[:,2]),
            "FLOOR", color='darkred', fontsize=12, ha='center')
    ax.text(np.mean(top_corners[:,0]), np.mean(top_corners[:,1]), np.mean(top_corners[:,2]),
            "CEILING", color='red', fontsize=12, ha='center')
    
    # Add a dummy plot for the legend
    ax.plot([0], [0], [0], color=color, alpha=0.5, label=label)

def transform_points(points, transform_matrix):
    """Transform a list of points using a transformation matrix"""
    transformed_points = []
    for point in points:
        # Add homogeneous coordinate
        point_h = np.append(point, 1)
        # Transform
        transformed_h = transform_matrix @ point_h
        # Remove homogeneous coordinate
        transformed_points.append(transformed_h[:3])
    return transformed_points

def plot_camera_volume(ax, camera_markers, height=0.6, color='blue', alpha=0.1, label="Camera Volume"):
    """
    Modified version for camera coordinates: Plot a 3D prism by computing the best-fit plane
    for the camera markers and extruding along its normal. This ensures that the ceiling in camera
    coordinates is a parallel translation of the base.
    """
    camera_markers = np.array(camera_markers)
    # Compute the centroid of the camera markers
    centroid = np.mean(camera_markers, axis=0)
    # Use SVD to compute the best-fit plane normal (direction of smallest variance)
    U, S, Vt = np.linalg.svd(camera_markers - centroid)
    normal = Vt[-1]
    
    # Project each marker onto the best-fit plane to get the consistent base
    bottom_corners = []
    for p in camera_markers:
        d = np.dot(p - centroid, normal)
        p_proj = p - d * normal
        bottom_corners.append(p_proj)
    bottom_corners = np.array(bottom_corners)
    
    # Compute the top face by extruding the base along the computed normal
    top_corners = bottom_corners + height * normal

    # Define the faces of the prism
    faces = []
    faces.append(bottom_corners.tolist())  # Bottom face
    faces.append(top_corners.tolist())       # Top face
    num_pts = len(bottom_corners)
    for i in range(num_pts):
        j = (i + 1) % num_pts
        face = [bottom_corners[i].tolist(), bottom_corners[j].tolist(),
                top_corners[j].tolist(), top_corners[i].tolist()]
        faces.append(face)
    
    # Plot each face with the same color and transparency
    for face in faces:
        poly = Poly3DCollection([face], alpha=alpha, facecolor=color, edgecolor=color, linewidth=1)
        ax.add_collection3d(poly)
    
    # Draw extrusion (vertical) lines for clarity
    for i in range(num_pts):
        ax.plot([bottom_corners[i][0], top_corners[i][0]],
                [bottom_corners[i][1], top_corners[i][1]],
                [bottom_corners[i][2], top_corners[i][2]], color=color, alpha=0.7)
    
    # Label the floor and ceiling (using average positions)
    ax.text(np.mean(bottom_corners[:,0]), np.mean(bottom_corners[:,1]), np.mean(bottom_corners[:,2]),
            "FLOOR", color='darkblue', fontsize=12, ha='center')
    ax.text(np.mean(top_corners[:,0]), np.mean(top_corners[:,1]), np.mean(top_corners[:,2]),
            "CEILING", color='blue', fontsize=12, ha='center')
    
    # Dummy plot for the legend
    ax.plot([0], [0], [0], color=color, alpha=0.5, label=label)

def main():
    print(f"Loading workspace file: {WORKSPACE_FILE}")
    print(f"Loading extrinsic file: {EXTRINSIC_FILE}")
    
    # Load workspace configuration
    workspace_data = load_yaml_file(WORKSPACE_FILE)
    if not workspace_data:
        print(f"Failed to load workspace file: {WORKSPACE_FILE}")
        return
    
    # Load extrinsic calibration
    bHc = load_extrinsic_file(EXTRINSIC_FILE)
    print(f"Camera to Robot Transform (bHc):\n{bHc}")
    
    # Calculate inverse transform - robot to camera
    cHb = np.linalg.inv(bHc)
    print(f"Robot to Camera Transform (cHb):\n{cHb}")
    
    # Extract marker positions in robot frame from workspace file
    robot_markers = []
    for marker in workspace_data['markers']:
        pos = marker['position']
        robot_markers.append(np.array([pos[0], pos[1], pos[2]]))
    
    # Get workspace boundaries
    bounds = workspace_data['workspace_boundaries']
    
    print("\nWorkspace Volume Analysis:")
    print(f"Z-min (floor): {bounds['z_min']:.4f} meters")
    print(f"Z-max (ceiling): {bounds['z_max']:.4f} meters")
    print("Marker Z-heights:")
    for i, marker in enumerate(robot_markers):
        print(f"  Marker {i}: {marker[2]:.4f} meters")
        
    if 'height_parameters' in workspace_data:
        height_params = workspace_data['height_parameters']
        print(f"Height below markers: {height_params['height_below_markers']:.4f} meters")
        print(f"Height above markers: {height_params['height_above_markers']:.4f} meters")
    
    # Convert marker positions to camera frame
    camera_markers = transform_points(robot_markers, cHb)
    
    # Create 3D plot for camera view
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    camera_origin = np.zeros(3)
    camera_R = np.eye(3)
    plot_coordinate_frame(ax, camera_origin, camera_R, scale=0.05, label_prefix="Camera-")
    
    robot_origin_in_camera = cHb[:3, 3]
    robot_R_in_camera = cHb[:3, :3]
    plot_coordinate_frame(ax, robot_origin_in_camera, robot_R_in_camera, scale=0.05, label_prefix="Robot-")
    
    plot_workspace(ax, camera_markers, color='blue', label_prefix="Camera-")
    
    # Use the modified camera volume function with best-fit plane extrusion
    plot_camera_volume(ax, camera_markers, height=0.6, color='blue', alpha=0.1, label="Camera Volume")
    
    # Include both camera markers and coordinate frame origins when calculating plot limits
    all_points = np.array(camera_markers + [camera_origin, robot_origin_in_camera])
    
    # Add some padding to ensure coordinate frames are fully visible
    x_min, y_min, z_min = np.min(all_points, axis=0) - 0.3
    x_max, y_max, z_max = np.max(all_points, axis=0) + 0.3
    
    max_range = max(x_max-x_min, y_max-y_min, z_max-z_min) / 3
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    mid_z = (z_min + z_max) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Workspace Visualization - Camera Coordinates')
    ax.legend()
    ax.view_init(elev=30, azim=45)
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.savefig('workspace_camera_view.png', dpi=200)
    print(f"Saved visualization to workspace_camera_view.png")
    
    # Create another figure for robot coordinate frame
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    robot_origin = np.zeros(3)
    robot_R = np.eye(3)
    plot_coordinate_frame(ax2, robot_origin, robot_R, scale=0.05, label_prefix="Robot-")
    
    camera_origin_in_robot = bHc[:3, 3]
    camera_R_in_robot = bHc[:3, :3]
    plot_coordinate_frame(ax2, camera_origin_in_robot, camera_R_in_robot, scale=0.05, label_prefix="Camera-")
    
    plot_workspace(ax2, robot_markers, color='red', label_prefix="Robot-")
    # Use the modified function to plot the volume with the ceiling generated by extruding along the best-fit plane normal.
    plot_workspace_volume(ax2, robot_markers, height=0.9, color='red')
    
    # Include both robot markers and coordinate frame origins when calculating plot limits
    all_points_robot = np.array(robot_markers + [robot_origin, camera_origin_in_robot])
    
    # Add padding to ensure coordinate frames are fully visible
    x_min_robot, y_min_robot, z_min_robot = np.min(all_points_robot, axis=0) - 0.3
    x_max_robot, y_max_robot, z_max_robot = np.max(all_points_robot, axis=0) + 0.3
    
    max_range_robot = max(x_max_robot-x_min_robot, y_max_robot-y_min_robot, z_max_robot-z_min_robot) / 3
    mid_x_robot = (x_min_robot + x_max_robot) / 2
    mid_y_robot = (y_min_robot + y_max_robot) / 2
    mid_z_robot = (z_min_robot + z_max_robot) / 2
    
    ax2.set_xlim(mid_x_robot - max_range_robot, mid_x_robot + max_range_robot)
    ax2.set_ylim(mid_y_robot - max_range_robot, mid_y_robot + max_range_robot)
    ax2.set_zlim(mid_z_robot - max_range_robot, mid_z_robot + max_range_robot)
    
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_zlabel('Z (meters)')
    ax2.set_title('Workspace Visualization - Robot Coordinates')
    ax2.legend()
    ax2.view_init(elev=30, azim=45)
    ax2.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.savefig('workspace_robot_view.png', dpi=200)
    print(f"Saved visualization to workspace_robot_view.png")
    
    print("Displaying plots (close window to exit)...")
    plt.show()

if __name__ == "__main__":
    main()




# #!/usr/bin/env python3
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import yaml
# import os
# import re

# # Hardcoded file paths - update these to match your actual paths
# WORKSPACE_FILE = "data/workspacenew.yaml"  # Changed from ../data to data
# EXTRINSIC_FILE = "data/test_extrinsic.yaml"  # Changed from ../data to data

# def parse_opencv_matrix(content):
#     """Parse OpenCV matrix format from YAML content"""
#     # Regular expression to find OpenCV matrices
#     matrix_pattern = r'(?:bHc|camera_matrix|dist_coeffs):\s*!!opencv-matrix\s*rows:\s*(\d+)\s*cols:\s*(\d+)\s*dt:\s*\w+\s*data:\s*\[(.*?)\]'
    
#     # Find all matrices in the content
#     matches = re.findall(matrix_pattern, content, re.DOTALL)
    
#     result = {}
#     for match in matches:
#         rows = int(match[0])
#         cols = int(match[1])
#         data_str = match[2].strip().replace('\n', ' ')
#         data_list = [float(x) for x in data_str.split(',')]
        
#         # Reshape into matrix
#         matrix = np.array(data_list).reshape(rows, cols)
        
#         # Find the name of this matrix
#         pattern = r'(bHc|camera_matrix|dist_coeffs):\s*!!opencv-matrix'
#         name_match = re.search(pattern, content)
#         if name_match:
#             name = name_match.group(1)
#             result[name] = matrix
    
#     return result

# def load_yaml_file(file_path):
#     """Load YAML file and return the data"""
#     try:
#         with open(file_path, 'r') as file:
#             content = file.read()
            
#             # Check if this is an OpenCV YAML file
#             if '!!opencv-matrix' in content:
#                 # Parse OpenCV matrices if present
#                 opencv_matrices = parse_opencv_matrix(content)
#                 if 'bHc' in opencv_matrices:
#                     return {'bHc': opencv_matrices['bHc']}
                
#             # Otherwise parse as regular YAML
#             if content.startswith('%YAML:1.0'):
#                 content = content.replace('%YAML:1.0', '')
#             return yaml.safe_load(content)
#     except Exception as e:
#         print(f"Error reading file: {e}")
#         return None

# def load_extrinsic_file(file_path):
#     """Load extrinsic calibration file and return the transformation matrix"""
#     data = load_yaml_file(file_path)
#     if data and 'bHc' in data:
#         bHc = np.array(data['bHc'], dtype=float)
#         return bHc
#     else:
#         # Fallback to default matrix if file can't be loaded
#         print("Warning: Could not load extrinsic calibration. Using default values.")
#         # Identity rotation, translation [0.3, 0, 0.5] based on your output
#         default_bHc = np.eye(4)
#         default_bHc[0:3, 3] = [0.3, 0, 0.5]
#         return default_bHc

# def plot_coordinate_frame(ax, origin, R, scale=0.1, label_prefix=""):
#     """Plot a coordinate frame with origin and rotation matrix R"""
#     colors = ['r', 'g', 'b']
#     labels = ['X', 'Y', 'Z']
    
#     # Plot coordinate axes
#     for i in range(3):
#         ax.quiver(origin[0], origin[1], origin[2],
#                  R[0, i]*scale, R[1, i]*scale, R[2, i]*scale,
#                  color=colors[i], label=f"{label_prefix}{labels[i]}")
        
#         # Add text labels
#         text_pos = origin + R[:3, i] * scale * 1.1
#         ax.text(text_pos[0], text_pos[1], text_pos[2], f"{label_prefix}{labels[i]}")

# def plot_workspace(ax, markers, color, label_prefix=""):
#     """Plot markers and connect them to form the workspace"""
#     # Extract x, y, z coordinates
#     xs = [marker[0] for marker in markers]
#     ys = [marker[1] for marker in markers]
#     zs = [marker[2] for marker in markers]
    
#     # Plot markers
#     ax.scatter(xs, ys, zs, color=color, s=100, marker='o', label=f"{label_prefix}Markers")
    
#     # Connect markers with lines to form the workspace bottom face
#     for i in range(len(markers)):
#         next_i = (i + 1) % len(markers)
#         ax.plot([xs[i], xs[next_i]], [ys[i], ys[next_i]], [zs[i], zs[next_i]], color=color)
    
#     # Add marker labels
#     for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
#         ax.text(x, y, z, f'{label_prefix}M{i}', color=color)

# # def plot_workspace_volume(ax, bounds, markers, color, alpha=0.1, label="Workspace Volume"):
# #     """Plot a 3D box representing the workspace volume with ceiling above markers"""
# #     x_min, x_max = bounds["x_min"], bounds["x_max"]
# #     y_min, y_max = bounds["y_min"], bounds["y_max"]
    
# #     # Use marker z-position as the floor
# #     floor_z = np.mean([marker[2] for marker in markers])
    
# #     # Set ceiling 0.6m above markers (not toward camera)
# #     ceiling_z = floor_z + 0.6  # 60cm above markers
    
# #     # Define the 8 corners of the box - with ceiling directly above floor
# #     corners = np.array([
# #         [x_min, y_min, floor_z],    # Bottom floor corner
# #         [x_max, y_min, floor_z],    # Bottom floor corner  
# #         [x_max, y_max, floor_z],    # Bottom floor corner
# #         [x_min, y_max, floor_z],    # Bottom floor corner
# #         [x_min, y_min, ceiling_z],  # Top ceiling corner (directly above floor)
# #         [x_max, y_min, ceiling_z],  # Top ceiling corner (directly above floor)
# #         [x_max, y_max, ceiling_z],  # Top ceiling corner (directly above floor)
# #         [x_min, y_max, ceiling_z]   # Top ceiling corner (directly above floor)
# #     ])
    
# #     # Define the 6 faces of the box using indices of corners
# #     faces = [
# #         [0, 1, 2, 3],  # Floor face (markers)
# #         [4, 5, 6, 7],  # Ceiling face (above markers)
# #         [0, 1, 5, 4],  # Side face
# #         [2, 3, 7, 6],  # Side face
# #         [0, 3, 7, 4],  # Side face
# #         [1, 2, 6, 5]   # Side face
# #     ]
    
# #     # Create a Poly3DCollection for the volume
# #     face_vertices = [[corners[idx] for idx in face] for face in faces]
    
# #     # Different colors for floor and ceiling
# #     face_colors = [color] * len(faces)
# #     face_colors[0] = 'darkred'  # Floor in darker red
# #     face_colors[1] = 'lightcoral'  # Ceiling in lighter red
    
# #     # Plot each face with its color
# #     for i, verts in enumerate(face_vertices):
# #         poly = Poly3DCollection([verts], alpha=alpha, facecolor=face_colors[i], edgecolor=face_colors[i], linewidth=1)
# #         ax.add_collection3d(poly)
    
# #     # Add vertical lines from markers to ceiling for clarity
# #     for marker in markers:
# #         # Line from marker to ceiling
# #         ax.plot([marker[0], marker[0]], 
# #                 [marker[1], marker[1]], 
# #                 [marker[2], ceiling_z], 
# #                 color='red', linestyle=':', alpha=0.7)
    
# #     # Mark the floor and ceiling planes
# #     center_x = (x_min + x_max) / 2
# #     center_y = (y_min + y_max) / 2
# #     ax.text(center_x, center_y, floor_z, "FLOOR", color='darkred', fontsize=12, ha='center')
# #     ax.text(center_x, center_y, ceiling_z, "CEILING", color='red', fontsize=12, ha='center')
    
# #     # Add a dummy plot for the legend
# #     ax.plot([0], [0], [0], color=color, alpha=0.5, label=label)
    
# #     # Print the workspace height information
# #     print(f"Ceiling height (0.6m above markers): {ceiling_z:.4f}m")
# #     print(f"Floor height (at markers): {floor_z:.4f}m")
# #     print(f"Workspace height: {ceiling_z - floor_z:.4f}m")

# def plot_workspace_volume(ax, bounds, markers, color, alpha=0.1, label="Workspace Volume"):
#     """Plot a 3D prism with the top face parallel to the base formed by markers."""
#     height = 0.6  # Height of the prism in meters
#     markers = np.array(markers)  # Convert markers to a NumPy array

#     # Step 1: Define the base plane using the average z-coordinate
#     floor_z = np.mean(markers[:, 2])  # Average height of markers
#     bottom_corners = markers.copy()
#     bottom_corners[:, 2] = floor_z  # Project markers onto the floor plane

#     # Step 2: Create the top face by shifting the base upward
#     top_corners = bottom_corners.copy()
#     top_corners[:, 2] += height  # Shift all points up by the height

#     # Step 3: Define the faces of the prism
#     faces = [
#         bottom_corners.tolist(),  # Bottom face (base)
#         top_corners.tolist(),     # Top face (ceiling)
#         # Side faces connecting corresponding corners
#         [bottom_corners[0].tolist(), bottom_corners[1].tolist(), top_corners[1].tolist(), top_corners[0].tolist()],  # Side 1
#         [bottom_corners[1].tolist(), bottom_corners[2].tolist(), top_corners[2].tolist(), top_corners[1].tolist()],  # Side 2
#         [bottom_corners[2].tolist(), bottom_corners[3].tolist(), top_corners[3].tolist(), top_corners[2].tolist()],  # Side 3
#         [bottom_corners[3].tolist(), bottom_corners[0].tolist(), top_corners[0].tolist(), top_corners[3].tolist()]   # Side 4
#     ]

#     # Plot each face
#     for face in faces:
#         poly = Poly3DCollection([face], alpha=alpha, facecolor=color, edgecolor=color, linewidth=1)
#         ax.add_collection3d(poly)

#     # Step 4: Draw vertical lines (sides) from bottom to top corners
#     for i in range(4):
#         ax.plot([bottom_corners[i][0], top_corners[i][0]],
#                 [bottom_corners[i][1], top_corners[i][1]],
#                 [bottom_corners[i][2], top_corners[i][2]], color=color, alpha=0.7)

#     # Add labels for clarity
#     center_x = np.mean(bottom_corners[:, 0])
#     center_y = np.mean(bottom_corners[:, 1])
#     ax.text(center_x, center_y, floor_z, "FLOOR", color='darkred', fontsize=12, ha='center')
#     ax.text(center_x, center_y, floor_z + height, "CEILING", color='red', fontsize=12, ha='center')

#     # Add a dummy plot for the legend
#     ax.plot([0], [0], [0], color=color, alpha=0.5, label=label)


# def transform_points(points, transform_matrix):
#     """Transform a list of points using a transformation matrix"""
#     transformed_points = []
#     for point in points:
#         # Add homogeneous coordinate
#         point_h = np.append(point, 1)
#         # Transform
#         transformed_h = transform_matrix @ point_h
#         # Remove homogeneous coordinate
#         transformed_points.append(transformed_h[:3])
#     return transformed_points

# def plot_camera_volume(ax, bounds, markers, camera_markers, cHb, color='blue', alpha=0.1):
#     """Plot workspace volume in camera coordinates"""
#     x_min, x_max = bounds["x_min"], bounds["x_max"]
#     y_min, y_max = bounds["y_min"], bounds["y_max"]
    
#     # Use marker z-position as the floor
#     floor_z = np.mean([marker[2] for marker in markers])
    
#     # Set ceiling 0.6m above markers (not toward camera)
#     ceiling_z = floor_z + 0.6  # 60cm above markers
    
#     # Define corners in robot frame
#     corners = np.array([
#         [x_min, y_min, floor_z],    # Floor corners
#         [x_max, y_min, floor_z],
#         [x_max, y_max, floor_z],
#         [x_min, y_max, floor_z],
#         [x_min, y_min, ceiling_z],  # Ceiling corners (directly above floor)
#         [x_max, y_min, ceiling_z],
#         [x_max, y_max, ceiling_z],
#         [x_min, y_max, ceiling_z]
#     ])
    
#     # Transform corners to camera frame
#     camera_corners = transform_points(corners, cHb)
    
#     # Define the 6 faces of the box
#     faces = [
#         [0, 1, 2, 3],  # Floor face (markers)
#         [4, 5, 6, 7],  # Ceiling face (above markers)
#         [0, 1, 5, 4],  # Side face
#         [2, 3, 7, 6],  # Side face
#         [0, 3, 7, 4],  # Side face
#         [1, 2, 6, 5]   # Side face
#     ]
    
#     # Create face vertices in camera frame
#     face_vertices = [[camera_corners[idx] for idx in face] for face in faces]
    
#     # Different colors for floor and ceiling
#     face_colors = [color] * len(faces)
#     face_colors[0] = 'darkblue'  # Floor in darker blue
#     face_colors[1] = 'lightskyblue'  # Ceiling in lighter blue
    
#     # Plot each face with its color
#     for i, verts in enumerate(face_vertices):
#         poly = Poly3DCollection([verts], alpha=alpha, facecolor=face_colors[i], edgecolor=face_colors[i], linewidth=1)
#         ax.add_collection3d(poly)
    
#     # Add vertical lines from camera markers to ceiling in camera frame
#     for i, marker in enumerate(camera_markers):
#         # Calculate ceiling point for this marker (0.6m above in robot frame)
#         ceiling_point_robot = [markers[i][0], markers[i][1], markers[i][2] + 0.6]
#         ceiling_point_camera = transform_points([ceiling_point_robot], cHb)[0]
        
#         # Line from marker to ceiling
#         ax.plot([marker[0], ceiling_point_camera[0]], 
#                 [marker[1], ceiling_point_camera[1]], 
#                 [marker[2], ceiling_point_camera[2]], 
#                 color='blue', linestyle=':', alpha=0.7)
    
#     # Mark the floor and ceiling planes in camera coordinates
#     floor_center = np.mean(camera_corners[:4], axis=0)
#     ceiling_center = np.mean(camera_corners[4:], axis=0)
#     ax.text(floor_center[0], floor_center[1], floor_center[2], "FLOOR", color='darkblue', fontsize=12, ha='center')
#     ax.text(ceiling_center[0], ceiling_center[1], ceiling_center[2], "CEILING", color='blue', fontsize=12, ha='center')
    
#     # Add a dummy plot for the legend
#     ax.plot([0], [0], [0], color=color, alpha=0.5, label="Camera-Volume")

# def main():
#     print(f"Loading workspace file: {WORKSPACE_FILE}")
#     print(f"Loading extrinsic file: {EXTRINSIC_FILE}")
    
#     # Load workspace configuration
#     workspace_data = load_yaml_file(WORKSPACE_FILE)
#     if not workspace_data:
#         print(f"Failed to load workspace file: {WORKSPACE_FILE}")
#         return
    
#     # Load extrinsic calibration
#     bHc = load_extrinsic_file(EXTRINSIC_FILE)
#     print(f"Camera to Robot Transform (bHc):\n{bHc}")
    
#     # Calculate inverse transform - robot to camera
#     cHb = np.linalg.inv(bHc)
#     print(f"Robot to Camera Transform (cHb):\n{cHb}")
    
#     # Extract marker positions in robot frame from workspace file
#     robot_markers = []
#     for marker in workspace_data['markers']:
#         pos = marker['position']
#         robot_markers.append(np.array([pos[0], pos[1], pos[2]]))
    
#     # Get workspace boundaries
#     bounds = workspace_data['workspace_boundaries']
    
#     # Print debug info about heights
#     print("\nWorkspace Volume Analysis:")
#     print(f"Z-min (floor): {bounds['z_min']:.4f} meters")
#     print(f"Z-max (ceiling): {bounds['z_max']:.4f} meters")
#     print("Marker Z-heights:")
#     for i, marker in enumerate(robot_markers):
#         print(f"  Marker {i}: {marker[2]:.4f} meters")
        
#     # If height parameters are available, print them
#     if 'height_parameters' in workspace_data:
#         height_params = workspace_data['height_parameters']
#         print(f"Height below markers: {height_params['height_below_markers']:.4f} meters")
#         print(f"Height above markers: {height_params['height_above_markers']:.4f} meters")
    
#     # Convert marker positions to camera frame
#     camera_markers = transform_points(robot_markers, cHb)
    
#     # Create 3D plot for camera view
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Plot camera coordinate frame - at origin in camera frame
#     camera_origin = np.zeros(3)
#     camera_R = np.eye(3)
#     plot_coordinate_frame(ax, camera_origin, camera_R, scale=0.05, label_prefix="Camera-")
    
#     # Plot robot coordinate frame in camera coordinates
#     robot_origin_in_camera = cHb[:3, 3]
#     robot_R_in_camera = cHb[:3, :3]
#     plot_coordinate_frame(ax, robot_origin_in_camera, robot_R_in_camera, scale=0.05, label_prefix="Robot-")
    
#     # Plot markers in camera frame
#     plot_workspace(ax, camera_markers, color='blue', label_prefix="Camera-")
    
#     # Plot workspace volume in camera frame
#     plot_camera_volume(ax, bounds, robot_markers, camera_markers, cHb)
    
#     # Calculate view boundaries based on workspace volume
#     all_points = np.array(camera_markers)
#     x_min, y_min, z_min = np.min(all_points, axis=0) - 0.2
#     x_max, y_max, z_max = np.max(all_points, axis=0) + 0.2
    
#     # Set plot limits for better visualization
#     max_range = max(x_max-x_min, y_max-y_min, z_max-z_min) / 2
#     mid_x = (x_min + x_max) / 2
#     mid_y = (y_min + y_max) / 2
#     mid_z = (z_min + z_max) / 2
    
#     ax.set_xlim(mid_x - max_range, mid_x + max_range)
#     ax.set_ylim(mid_y - max_range, mid_y + max_range)
#     ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
#     # Set plot labels and title
#     ax.set_xlabel('X (meters)')
#     ax.set_ylabel('Y (meters)')
#     ax.set_zlabel('Z (meters)')
#     ax.set_title('Workspace Visualization - Camera Coordinates')
    
#     # Add legend
#     ax.legend()
    
#     # Adjust view for better visualization
#     ax.view_init(elev=30, azim=45)
    
#     # Set equal aspect ratio
#     ax.set_box_aspect([1, 1, 1])
    
#     plt.tight_layout()
    
#     # Save figure
#     plt.savefig('workspace_camera_view.png', dpi=200)
#     print(f"Saved visualization to workspace_camera_view.png")
    
#     # Create another figure for robot coordinate frame
#     fig2 = plt.figure(figsize=(12, 10))
#     ax2 = fig2.add_subplot(111, projection='3d')
    
#     # Plot robot coordinate frame - at origin in robot frame
#     robot_origin = np.zeros(3)
#     robot_R = np.eye(3)
#     plot_coordinate_frame(ax2, robot_origin, robot_R, scale=0.05, label_prefix="Robot-")
    
#     # Plot camera coordinate frame in robot coordinates
#     camera_origin_in_robot = bHc[:3, 3]
#     camera_R_in_robot = bHc[:3, :3]
#     plot_coordinate_frame(ax2, camera_origin_in_robot, camera_R_in_robot, scale=0.05, label_prefix="Camera-")
    
#     # Plot markers in robot frame
#     plot_workspace(ax2, robot_markers, color='red', label_prefix="Robot-")
    
#     # Plot workspace volume in robot frame - NOW CORRECTLY POSITIONED
#     camera_position = bHc[:3, 3]  # Extract camera position from transform
#     # plot_workspace_volume(ax2, bounds, robot_markers, camera_position, color='red')
#     plot_workspace_volume(ax2, bounds, robot_markers, color='red')
    
#     # Set plot limits for better visualization
#     margin = 0.1
#     x_range = bounds['x_max'] - bounds['x_min']
#     y_range = bounds['y_max'] - bounds['y_min']
#     z_range = bounds['z_max'] - bounds['z_min']
    
#     x_center = (bounds['x_max'] + bounds['x_min']) / 2
#     y_center = (bounds['y_max'] + bounds['y_min']) / 2
#     z_center = (bounds['z_max'] + bounds['z_min']) / 2
    
#     max_range = max(x_range, y_range, z_range) / 2
    
#     ax2.set_xlim(x_center - max_range * 1.5, x_center + max_range * 1.5)
#     ax2.set_ylim(y_center - max_range * 1.5, y_center + max_range * 1.5)
#     ax2.set_zlim(bounds['z_min'] - 0.1, bounds['z_max'] + 0.1)  # Focus on the workspace z-range
    
#     # Set plot labels and title
#     ax2.set_xlabel('X (meters)')
#     ax2.set_ylabel('Y (meters)')
#     ax2.set_zlabel('Z (meters)')
#     ax2.set_title('Workspace Visualization - Robot Coordinates')
    
#     # Add legend
#     ax2.legend()
    
#     # Adjust view for better visualization
#     ax2.view_init(elev=30, azim=45)
    
#     # Set equal aspect ratio
#     ax2.set_box_aspect([1, 1, 1])
    
#     plt.tight_layout()
    
#     # Save figure
#     plt.savefig('workspace_robot_view.png', dpi=200)
#     print(f"Saved visualization to workspace_robot_view.png")
    
#     # Show plots
#     print("Displaying plots (close window to exit)...")
#     plt.show()

# if __name__ == "__main__":
#     main()
