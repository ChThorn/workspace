#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import re
import os

# File paths - change these to match your files
WORKSPACE_YAML = "data/workspacenew.yaml"  # Path to your workspace YAML file
EXTRINSIC_YAML = "data/test_extrinsic.yaml"  # Path to your extrinsic calibration file

# Workspace extension parameters (in meters)
HEIGHT_ABOVE = 0.6  # Height above each marker (toward camera)
HEIGHT_BELOW = 0.03  # Height below each marker (away from camera)

def parse_opencv_yaml(file_path):
    """
    Parse OpenCV YAML file with robust error handling
    """
    print(f"Reading file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Remove the YAML directive
        content = re.sub(r'%YAML:[\d\.]+', '', content)
        
        # Check if this is a workspace YAML
        if 'workspace_boundaries' in content:
            result = {}
            
            # Extract markers
            markers = []
            marker_pattern = r'id:(\d+),\s*position:\s*\[\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*\]'
            
            for match in re.finditer(marker_pattern, content):
                marker_id = int(match.group(1))
                x = float(match.group(2))
                y = float(match.group(3))
                z = float(match.group(4))
                markers.append({
                    'id': marker_id,
                    'position': [x, y, z]
                })
            
            result['markers'] = markers
            
            print(f"Successfully parsed workspace file with {len(markers)} markers")
            return result
        
        # Check if this is an extrinsic matrix YAML
        elif 'bHc' in content:
            # Try to extract the 4x4 matrix values
            # First, look for a data block with 16 values
            data_pattern = r'data:\s*\[(.*?)\]'
            data_match = re.search(data_pattern, content, re.DOTALL)
            
            if data_match:
                data_text = data_match.group(1).replace('\n', ' ')
                values = re.findall(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', data_text)
                
                if len(values) == 16:  # 4x4 matrix
                    matrix = np.array([float(v) for v in values]).reshape(4, 4)
                    print(f"Successfully parsed extrinsic matrix:")
                    print(matrix)
                    return matrix
            
            # If we couldn't parse it, use a fallback
            print("Warning: Could not parse extrinsic matrix format, using fallback")
            return np.array([
                [1.0, 0.0, 0.0, 0.3],
                [0.0, 0.7071, -0.7071, 0.0],
                [0.0, 0.7071, 0.7071, 0.5],
                [0.0, 0.0, 0.0, 1.0]
            ])
        
        # Unknown file format
        print(f"Warning: Unknown file format in {file_path}")
        return None
        
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None

def transform_point_robot_to_camera(point, extrinsic_matrix):
    """Transform a point from robot coordinates to camera coordinates"""
    # Homogeneous point
    point_h = np.append(point, 1.0)
    
    # Inverse of extrinsic matrix (robot to camera transform)
    camera_from_robot = np.linalg.inv(extrinsic_matrix)
    
    # Transform point
    point_camera = camera_from_robot.dot(point_h)
    
    # Return 3D point
    return point_camera[:3]

def create_marker_vertices(center, size=0.02):
    """Create vertices for a marker cube centered at the given point"""
    half_size = size / 2
    vertices = np.array([
        [center[0] - half_size, center[1] - half_size, center[2] - half_size],  # 0
        [center[0] + half_size, center[1] - half_size, center[2] - half_size],  # 1
        [center[0] + half_size, center[1] + half_size, center[2] - half_size],  # 2
        [center[0] - half_size, center[1] + half_size, center[2] - half_size],  # 3
        [center[0] - half_size, center[1] - half_size, center[2] + half_size],  # 4
        [center[0] + half_size, center[1] - half_size, center[2] + half_size],  # 5
        [center[0] + half_size, center[1] + half_size, center[2] + half_size],  # 6
        [center[0] - half_size, center[1] + half_size, center[2] + half_size]   # 7
    ])
    return vertices

def create_workspace_with_per_marker_offsets(marker_positions_camera, height_above=HEIGHT_ABOVE, height_below=HEIGHT_BELOW):
    """Create a workspace volume where each marker has the exact same offset above and below"""
    # Calculate the X/Y coordinates of markers (for the rectangular outline)
    # Sort markers to form a proper rectangle (using convex hull approach)
    markers_xy = marker_positions_camera[:, :2]  # Just use X,Y coordinates
    center = np.mean(markers_xy, axis=0)
    
    # Sort markers by angle around center
    angles = np.arctan2(markers_xy[:, 1] - center[1], markers_xy[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_markers = marker_positions_camera[sorted_indices]
    
    # Create points exactly height_below below each marker
    floor_vertices = np.array([
        [sorted_markers[0, 0], sorted_markers[0, 1], sorted_markers[0, 2] + height_below],
        [sorted_markers[1, 0], sorted_markers[1, 1], sorted_markers[1, 2] + height_below],
        [sorted_markers[2, 0], sorted_markers[2, 1], sorted_markers[2, 2] + height_below],
        [sorted_markers[3, 0], sorted_markers[3, 1], sorted_markers[3, 2] + height_below]
    ])
    
    # Create points exactly height_above above each marker
    ceiling_vertices = np.array([
        [sorted_markers[0, 0], sorted_markers[0, 1], sorted_markers[0, 2] - height_above],
        [sorted_markers[1, 0], sorted_markers[1, 1], sorted_markers[1, 2] - height_above],
        [sorted_markers[2, 0], sorted_markers[2, 1], sorted_markers[2, 2] - height_above],
        [sorted_markers[3, 0], sorted_markers[3, 1], sorted_markers[3, 2] - height_above]
    ])
    
    # Calculate min/max for reporting
    min_x = np.min(sorted_markers[:, 0])
    max_x = np.max(sorted_markers[:, 0])
    min_y = np.min(sorted_markers[:, 1])
    max_y = np.max(sorted_markers[:, 1])
    
    min_ceiling_z = np.min(ceiling_vertices[:, 2])
    max_ceiling_z = np.max(ceiling_vertices[:, 2])
    min_floor_z = np.min(floor_vertices[:, 2])
    max_floor_z = np.max(floor_vertices[:, 2])
    
    print(f"Created workspace with per-marker offsets:")
    print(f"  X: [{min_x:.4f}, {max_x:.4f}]")
    print(f"  Y: [{min_y:.4f}, {max_y:.4f}]")
    print(f"  Z range (ceiling): [{min_ceiling_z:.4f}, {max_ceiling_z:.4f}]")
    print(f"  Z range (markers): [{np.min(sorted_markers[:, 2]):.4f}, {np.max(sorted_markers[:, 2]):.4f}]")
    print(f"  Z range (floor): [{min_floor_z:.4f}, {max_floor_z:.4f}]")
    print(f"  Exact height above each marker: {height_above:.2f}m")
    print(f"  Exact height below each marker: {height_below:.2f}m")
    
    # Combine all vertices for convenience (though we'll draw each face separately)
    all_vertices = np.vstack([sorted_markers, ceiling_vertices, floor_vertices])
    
    return sorted_indices, all_vertices

def plot_workspace_camera_coords():
    """Plot the workspace in camera coordinates, reading data from files"""
    # Check if files exist
    if not os.path.exists(WORKSPACE_YAML):
        print(f"Error: Workspace file {WORKSPACE_YAML} not found")
        return
    
    if not os.path.exists(EXTRINSIC_YAML):
        print(f"Error: Extrinsic file {EXTRINSIC_YAML} not found")
        return
    
    # Parse the files
    workspace_data = parse_opencv_yaml(WORKSPACE_YAML)
    extrinsic_matrix = parse_opencv_yaml(EXTRINSIC_YAML)
    
    if workspace_data is None or extrinsic_matrix is None:
        print("Error: Failed to parse one or both files")
        return
    
    # Get the camera-from-robot transform
    camera_from_robot = np.linalg.inv(extrinsic_matrix)
    print(f"Camera-from-Robot Transform (cHb):\n{camera_from_robot}")
    
    # Extract marker data
    markers = workspace_data['markers']
    marker_positions_robot = np.array([marker['position'] for marker in markers])
    marker_ids = [marker['id'] for marker in markers]
    
    # Transform marker positions to camera coordinates
    marker_positions_camera = np.array([
        transform_point_robot_to_camera(pos, extrinsic_matrix) for pos in marker_positions_robot
    ])
    
    # Print marker positions in camera coordinates
    print("\nMarker positions in camera coordinates:")
    for i, (pos, id) in enumerate(zip(marker_positions_camera, marker_ids)):
        print(f"  Marker {id}: {pos}")
    
    # Create workspace vertices with per-marker offsets
    sorted_indices, all_vertices = create_workspace_with_per_marker_offsets(marker_positions_camera)
    
    # Create reordered marker positions and IDs
    sorted_markers = marker_positions_camera[sorted_indices]
    sorted_ids = [marker_ids[i] for i in sorted_indices]
    
    # Extract the vertex groups from the all_vertices array
    marker_vertices = all_vertices[:4]  # First 4 are the marker positions
    ceiling_vertices = all_vertices[4:8]  # Next 4 are the ceiling vertices
    floor_vertices = all_vertices[8:12]  # Last 4 are the floor vertices
    
    # Calculate overall min/max values for all elements
    all_points = np.vstack([
        all_vertices,
        marker_positions_camera,
        np.array([[0, 0, 0]])  # camera origin
    ])
    
    min_x, max_x = np.min(all_points[:, 0]) - 0.05, np.max(all_points[:, 0]) + 0.05
    min_y, max_y = np.min(all_points[:, 1]) - 0.05, np.max(all_points[:, 1]) + 0.05
    min_z, max_z = np.min(all_points[:, 2]) - 0.05, np.max(all_points[:, 2]) + 0.05
    
    print(f"\nOverall coordinate ranges (with margin):")
    print(f"  X: [{min_x:.4f}, {max_x:.4f}]")
    print(f"  Y: [{min_y:.4f}, {max_y:.4f}]")
    print(f"  Z: [{min_z:.4f}, {max_z:.4f}]")
    
    # Set up the figure and axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define the faces
    faces = []
    
    # Add the ceiling face (slightly uneven due to marker height differences)
    faces.append([ceiling_vertices[0], ceiling_vertices[1], ceiling_vertices[2], ceiling_vertices[3]])
    
    # Add the floor face (slightly uneven due to marker height differences)
    faces.append([floor_vertices[0], floor_vertices[1], floor_vertices[2], floor_vertices[3]])
    
    # Add side faces connecting the original markers to the ceiling
    for i in range(4):
        j = (i + 1) % 4  # Next vertex index
        faces.append([marker_vertices[i], marker_vertices[j], ceiling_vertices[j], ceiling_vertices[i]])
    
    # Add side faces connecting the original markers to the floor
    for i in range(4):
        j = (i + 1) % 4  # Next vertex index
        faces.append([marker_vertices[i], marker_vertices[j], floor_vertices[j], floor_vertices[i]])
    
    # Create the workspace with all faces
    workspace = Poly3DCollection(faces, alpha=0.35, linewidths=1.5, edgecolor='darkblue')
    workspace.set_facecolor('cyan')
    ax.add_collection3d(workspace)
    
    # Draw connecting lines to make the structure clearer
    for i in range(4):
        # Connect marker to its ceiling point
        ax.plot([marker_vertices[i, 0], ceiling_vertices[i, 0]],
                [marker_vertices[i, 1], ceiling_vertices[i, 1]],
                [marker_vertices[i, 2], ceiling_vertices[i, 2]],
                'b-', alpha=0.5, linewidth=1)
        
        # Connect marker to its floor point
        ax.plot([marker_vertices[i, 0], floor_vertices[i, 0]],
                [marker_vertices[i, 1], floor_vertices[i, 1]],
                [marker_vertices[i, 2], floor_vertices[i, 2]],
                'b-', alpha=0.5, linewidth=1)
    
    # Draw ceiling outline
    ceiling_loop = np.vstack([ceiling_vertices, ceiling_vertices[0]])
    ax.plot(ceiling_loop[:, 0], ceiling_loop[:, 1], ceiling_loop[:, 2], 'b-', alpha=0.7, linewidth=1.5)
    
    # Draw floor outline
    floor_loop = np.vstack([floor_vertices, floor_vertices[0]])
    ax.plot(floor_loop[:, 0], floor_loop[:, 1], floor_loop[:, 2], 'b-', alpha=0.7, linewidth=1.5)
    
    # ----- Plot the markers as cubes -----
    marker_colors = ['orange', 'gold', 'darkorange', 'orangered']
    
    for i, (pos, id) in enumerate(zip(marker_positions_camera, marker_ids)):
        # Create small cube for each marker
        marker_vertices = create_marker_vertices(pos, size=0.02)
        
        # Define the faces of the cube
        marker_faces = [
            [marker_vertices[0], marker_vertices[1], marker_vertices[2], marker_vertices[3]],  # Bottom
            [marker_vertices[4], marker_vertices[5], marker_vertices[6], marker_vertices[7]],  # Top
            [marker_vertices[0], marker_vertices[1], marker_vertices[5], marker_vertices[4]],  # Front
            [marker_vertices[2], marker_vertices[3], marker_vertices[7], marker_vertices[6]],  # Back
            [marker_vertices[0], marker_vertices[3], marker_vertices[7], marker_vertices[4]],  # Left
            [marker_vertices[1], marker_vertices[2], marker_vertices[6], marker_vertices[5]]   # Right
        ]
        
        # Create marker cube
        marker_cube = Poly3DCollection(marker_faces, alpha=0.7, linewidths=1, edgecolor='k')
        marker_cube.set_facecolor(marker_colors[i % len(marker_colors)])
        ax.add_collection3d(marker_cube)
        
        # Add marker label
        ax.text(pos[0], pos[1], pos[2], f"  Marker {id}", fontsize=10)
        
    # ----- Plot the marker table (connecting the markers) -----
    # Use sorted markers to draw the table outline
    table_points = np.vstack([sorted_markers, sorted_markers[0]])  # Close the loop
    
    # Draw the table outline
    ax.plot(table_points[:, 0], table_points[:, 1], table_points[:, 2], 'r-', linewidth=2, label="Marker Table")
    
    # ----- Plot camera at origin -----
    camera_pos = np.array([0, 0, 0])  # Camera is at origin in camera coordinates
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], c='purple', marker='^', s=200, label='Camera')
    ax.text(camera_pos[0], camera_pos[1], camera_pos[2], "  Camera", fontsize=12)
    
    # ----- Draw camera coordinate system axes -----
    axis_length = 0.15
    # X-axis (red)
    ax.plot([0, axis_length], [0, 0], [0, 0], 'r-', linewidth=2)
    ax.text(axis_length*1.1, 0, 0, "X", color='red', fontsize=12)
    # Y-axis (green)
    ax.plot([0, 0], [0, axis_length], [0, 0], 'g-', linewidth=2)
    ax.text(0, axis_length*1.1, 0, "Y", color='green', fontsize=12)
    # Z-axis (blue)
    ax.plot([0, 0], [0, 0], [0, axis_length], 'b-', linewidth=2)
    ax.text(0, 0, axis_length*1.1, "Z", color='blue', fontsize=12)
    
    # ----- Set labels and title -----
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Workspace in Camera Coordinates with Marker Table')
    
    # ----- Create legend -----
    cuboid_patch = mpatches.Patch(color='cyan', alpha=0.35, label='Workspace Volume')
    marker_patch = mpatches.Patch(color='orange', alpha=0.7, label='ArUco Markers')
    table_line = mlines.Line2D([], [], color='red', linewidth=2, label='Marker Table')
    purple_triangle = mlines.Line2D([], [], color='purple', marker='^', linestyle='None',
                                  markersize=10, label='Camera')
    red_line = mlines.Line2D([], [], color='red', linewidth=2, label='X-axis')
    green_line = mlines.Line2D([], [], color='green', linewidth=2, label='Y-axis')
    blue_line = mlines.Line2D([], [], color='blue', linewidth=2, label='Z-axis')
    
    # Add legend
    ax.legend(handles=[cuboid_patch, marker_patch, table_line, purple_triangle, 
                       red_line, green_line, blue_line],
             loc='upper right')
    
    # ----- Set axis limits -----
    # Use the calculated bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    
    # ----- Set initial view -----
    # Try a different view angle to see the full workspace
    ax.view_init(elev=30, azim=50)
    
    # ----- Display note -----
    note_text = (f"Note: Each marker has exactly {HEIGHT_ABOVE:.2f}m space above it and {HEIGHT_BELOW:.2f}m below it.\n"
                f"This creates a workspace that follows the individual marker heights.")
    fig.text(0.5, 0.01, note_text, ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_workspace_camera_coords()