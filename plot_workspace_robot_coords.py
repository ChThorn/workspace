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

def parse_opencv_yaml(file_path):
    """Parse OpenCV YAML file with robust error handling"""
    print(f"Reading file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Remove the YAML directive
        content = re.sub(r'%YAML:[\d\.]+', '', content)
        
        # Check if this is a workspace YAML
        if 'workspace_boundaries' in content:
            result = {}
            
            # Extract workspace boundaries
            x_min_pattern = r'x_min\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
            x_max_pattern = r'x_max\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
            y_min_pattern = r'y_min\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
            y_max_pattern = r'y_max\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
            z_min_pattern = r'z_min\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
            z_max_pattern = r'z_max\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
            
            # Try to parse boundaries
            x_min_match = re.search(x_min_pattern, content)
            x_max_match = re.search(x_max_pattern, content)
            y_min_match = re.search(y_min_pattern, content)
            y_max_match = re.search(y_max_pattern, content)
            z_min_match = re.search(z_min_pattern, content)
            z_max_match = re.search(z_max_pattern, content)
            
            if all([x_min_match, x_max_match, y_min_match, y_max_match, z_min_match, z_max_match]):
                result['boundaries'] = {
                    'x_min': float(x_min_match.group(1)),
                    'x_max': float(x_max_match.group(1)),
                    'y_min': float(y_min_match.group(1)),
                    'y_max': float(y_max_match.group(1)),
                    'z_min': float(z_min_match.group(1)),
                    'z_max': float(z_max_match.group(1))
                }
                print("Successfully parsed workspace boundaries:")
                for key, value in result['boundaries'].items():
                    print(f"  {key}: {value:.4f}")
            else:
                print("Warning: Could not parse workspace boundaries")
            
            # Extract height parameters
            height_above_pattern = r'height_above_markers\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
            height_below_pattern = r'height_below_markers\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
            
            height_above_match = re.search(height_above_pattern, content)
            height_below_match = re.search(height_below_pattern, content)
            
            if height_above_match and height_below_match:
                result['height_params'] = {
                    'height_above': float(height_above_match.group(1)),
                    'height_below': float(height_below_match.group(1))
                }
                print(f"Found height parameters: above={result['height_params']['height_above']}m, below={result['height_params']['height_below']}m")
            
            # Extract markers
            markers = []
            float_pattern = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
            marker_pattern = r'id:(\d+),\s*position:\s*\[\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*\]'
            
            for match in re.finditer(marker_pattern, content):
                marker_id = int(match.group(1))
                x = float(match.group(2))
                y = float(match.group(3))
                z = float(match.group(4))
                
                marker = {'id': marker_id, 'position': [x, y, z]}
                
                # Look for ceiling and floor points
                ceiling_pattern = f'id:{marker_id}.*?ceiling.*?\\[\\s*({float_pattern}),\\s*({float_pattern}),\\s*({float_pattern})\\s*\\]'
                floor_pattern = f'id:{marker_id}.*?floor.*?\\[\\s*({float_pattern}),\\s*({float_pattern}),\\s*({float_pattern})\\s*\\]'
                
                ceiling_match = re.search(ceiling_pattern, content, re.DOTALL)
                if ceiling_match:
                    marker['ceiling'] = [float(ceiling_match.group(1)), float(ceiling_match.group(2)), float(ceiling_match.group(3))]
                
                floor_match = re.search(floor_pattern, content, re.DOTALL)
                if floor_match:
                    marker['floor'] = [float(floor_match.group(1)), float(floor_match.group(2)), float(floor_match.group(3))]
                
                markers.append(marker)
            
            result['markers'] = markers
            print(f"Successfully parsed workspace file with {len(markers)} markers")
            return result
        
        # Check if this is an extrinsic matrix YAML
        elif 'bHc' in content:
            # Try to extract the 4x4 matrix values
            data_pattern = r'data:\s*\[(.*?)\]'
            data_match = re.search(data_pattern, content, re.DOTALL)
            
            if data_match:
                data_text = data_match.group(1).replace('\n', ' ')
                values = re.findall(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', data_text)
                
                if len(values) == 16:  # 4x4 matrix
                    matrix = np.array([float(v) for v in values]).reshape(4, 4)
                    print(f"Successfully parsed extrinsic matrix")
                    return matrix
            
            # Fallback
            print("Warning: Could not parse extrinsic matrix format, using fallback")
            return np.array([
                [1.0, 0.0, 0.0, 0.3],
                [0.0, 0.7071, -0.7071, 0.0],
                [0.0, 0.7071, 0.7071, 0.5],
                [0.0, 0.0, 0.0, 1.0]
            ])
        
        print(f"Warning: Unknown file format in {file_path}")
        return None
        
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None

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

def plot_workspace_robot_coords():
    """Plot the workspace in robot coordinates, reading data from files"""
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
    
    # Get camera position in robot coordinates (from extrinsic matrix)
    camera_pos_robot = extrinsic_matrix[:3, 3]
    print(f"Camera position in robot coordinates: {camera_pos_robot}")
    
    # Extract marker data in robot coordinates
    markers = workspace_data['markers']
    marker_positions_robot = np.array([marker['position'] for marker in markers])
    marker_ids = [marker['id'] for marker in markers]

    # Get height parameters (if available)
    height_params = workspace_data.get('height_params', {})
    # height_above = height_params.get('height_above', 0.6)  # Default if not found
    # height_below = height_params.get('height_below', 0.03)  # Default if not found
    
    height_above = height_params['height_above']  # No default
    height_below = height_params['height_below']  # No default
    
    # Print marker positions in robot coordinates
    print("\nMarker positions in robot coordinates:")
    for i, (pos, id) in enumerate(zip(marker_positions_robot, marker_ids)):
        print(f"  Marker {id}: {pos}")
    
    # Extract or calculate ceiling and floor points
    ceiling_points = []
    floor_points = []
    
    for marker in markers:
        if 'ceiling' in marker:
            ceiling_points.append(marker['ceiling'])
        else:
            # Fallback calculation if not in YAML
            pos = marker['position']
            ceiling_points.append([pos[0], pos[1], pos[2] + height_above])
            
        if 'floor' in marker:
            floor_points.append(marker['floor'])
        else:
            # Fallback calculation if not in YAML
            pos = marker['position']
            floor_points.append([pos[0], pos[1], pos[2] - height_below])
    
    ceiling_points = np.array(ceiling_points)
    floor_points = np.array(floor_points)
    
    # Sort markers by angle around center (for proper rectangular outline)
    markers_xy = marker_positions_robot[:, :2]
    center = np.mean(markers_xy, axis=0)
    angles = np.arctan2(markers_xy[:, 1] - center[1], markers_xy[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    
    # Apply sorting to all arrays
    sorted_markers = marker_positions_robot[sorted_indices]
    sorted_ids = [marker_ids[i] for i in sorted_indices]
    sorted_ceiling = ceiling_points[sorted_indices]
    sorted_floor = floor_points[sorted_indices]
    
    # Calculate overall min/max values for all elements
    all_points = np.vstack([
        sorted_markers,
        sorted_ceiling,
        sorted_floor,
        camera_pos_robot.reshape(1, 3),  # Camera position
        np.array([[0, 0, 0]])  # Robot origin
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
    
    # Define the faces for workspace volume
    faces = []
    
    # Add the ceiling face
    faces.append([sorted_ceiling[0], sorted_ceiling[1], sorted_ceiling[2], sorted_ceiling[3]])
    
    # Add the floor face
    faces.append([sorted_floor[0], sorted_floor[1], sorted_floor[2], sorted_floor[3]])
    
    # Add side faces connecting markers to ceiling
    for i in range(4):
        j = (i + 1) % 4  # Next vertex index
        faces.append([sorted_markers[i], sorted_markers[j], sorted_ceiling[j], sorted_ceiling[i]])
    
    # Add side faces connecting markers to floor
    for i in range(4):
        j = (i + 1) % 4  # Next vertex index
        faces.append([sorted_markers[i], sorted_markers[j], sorted_floor[j], sorted_floor[i]])
    
    # Create the workspace with all faces
    workspace = Poly3DCollection(faces, alpha=0.35, linewidths=1.5, edgecolor='darkblue')
    workspace.set_facecolor('cyan')
    ax.add_collection3d(workspace)
    
    # Draw connecting lines to make the structure clearer
    for i in range(4):
        # Connect marker to its ceiling point
        ax.plot([sorted_markers[i, 0], sorted_ceiling[i, 0]],
                [sorted_markers[i, 1], sorted_ceiling[i, 1]],
                [sorted_markers[i, 2], sorted_ceiling[i, 2]],
                'b-', alpha=0.5, linewidth=1)
        
        # Connect marker to its floor point
        ax.plot([sorted_markers[i, 0], sorted_floor[i, 0]],
                [sorted_markers[i, 1], sorted_floor[i, 1]],
                [sorted_markers[i, 2], sorted_floor[i, 2]],
                'b-', alpha=0.5, linewidth=1)
    
    # Draw ceiling outline
    ceiling_loop = np.vstack([sorted_ceiling, sorted_ceiling[0]])
    ax.plot(ceiling_loop[:, 0], ceiling_loop[:, 1], ceiling_loop[:, 2], 'b-', alpha=0.7, linewidth=1.5)
    
    # Draw floor outline
    floor_loop = np.vstack([sorted_floor, sorted_floor[0]])
    ax.plot(floor_loop[:, 0], floor_loop[:, 1], floor_loop[:, 2], 'b-', alpha=0.7, linewidth=1.5)
    
    # Plot the markers as cubes
    marker_colors = ['orange', 'gold', 'darkorange', 'orangered']
    
    for i, (pos, id) in enumerate(zip(marker_positions_robot, marker_ids)):
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
    
    # Plot the marker table (connecting the markers)
    table_points = np.vstack([sorted_markers, sorted_markers[0]])  # Close the loop
    ax.plot(table_points[:, 0], table_points[:, 1], table_points[:, 2], 'r-', linewidth=2, label="Marker Table")
    
    # Plot camera position
    ax.scatter(camera_pos_robot[0], camera_pos_robot[1], camera_pos_robot[2], 
              c='purple', marker='^', s=200, label='Camera')
    ax.text(camera_pos_robot[0], camera_pos_robot[1], camera_pos_robot[2], 
           "  Camera", fontsize=12)
    
    # Draw camera viewing direction
    marker_center = np.mean(marker_positions_robot, axis=0)
    direction = marker_center - camera_pos_robot
    direction = direction / np.linalg.norm(direction) * 0.2  # Normalize and scale
    
    ax.plot([camera_pos_robot[0], camera_pos_robot[0] + direction[0]],
            [camera_pos_robot[1], camera_pos_robot[1] + direction[1]],
            [camera_pos_robot[2], camera_pos_robot[2] + direction[2]],
            'purple', linestyle='--', linewidth=2, alpha=0.7)
    
    # Draw robot coordinate system axes
    origin = np.array([0, 0, 0])
    axis_length = 0.15
    
    # X-axis (red)
    ax.plot([origin[0], origin[0] + axis_length], 
            [origin[1], origin[1]], 
            [origin[2], origin[2]], 'r-', linewidth=2)
    ax.text(origin[0] + axis_length*1.1, origin[1], origin[2], "X", color='red', fontsize=12)
    
    # Y-axis (green)
    ax.plot([origin[0], origin[0]], 
            [origin[1], origin[1] + axis_length], 
            [origin[2], origin[2]], 'g-', linewidth=2)
    ax.text(origin[0], origin[1] + axis_length*1.1, origin[2], "Y", color='green', fontsize=12)
    
    # Z-axis (blue)
    ax.plot([origin[0], origin[0]], 
            [origin[1], origin[1]], 
            [origin[2], origin[2] + axis_length], 'b-', linewidth=2)
    ax.text(origin[0], origin[1], origin[2] + axis_length*1.1, "Z", color='blue', fontsize=12)
    
    # Add a text label for the robot origin
    ax.text(origin[0], origin[1], origin[2], "  Robot Origin", fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Workspace in Robot Coordinates with Marker Table')
    
    # Create legend
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
    
    # Set axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    
    # Set initial view
    ax.view_init(elev=30, azim=120)
    
    # Display note
    note_text = (f"Note: In robot coordinates, the camera is at {camera_pos_robot}.\n"
                f"Each marker has {height_above:.2f}m above it and {height_below:.2f}m below it.")
    fig.text(0.5, 0.01, note_text, ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_workspace_robot_coords()


# #!/usr/bin/env python3
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import matplotlib.patches as mpatches
# import matplotlib.lines as mlines
# import re
# import os

# # File paths - change these to match your files
# WORKSPACE_YAML = "data/workspacenew.yaml"  # Path to your workspace YAML file
# EXTRINSIC_YAML = "data/test_extrinsic.yaml"  # Path to your extrinsic calibration file

# def parse_opencv_yaml(file_path):
#     """
#     Parse OpenCV YAML file with robust error handling
#     """
#     print(f"Reading file: {file_path}")
#     try:
#         with open(file_path, 'r') as f:
#             content = f.read()
            
#         # Remove the YAML directive
#         content = re.sub(r'%YAML:[\d\.]+', '', content)
        
#         # Check if this is a workspace YAML
#         if 'workspace_boundaries' in content:
#             result = {}
            
#             # Extract workspace boundaries using a more flexible approach
#             # Try multiple regex patterns to account for different YAML formats
            
#             # Pattern 1: Standard format with colons
#             bounds_pattern1 = r'workspace_boundaries\s*:\s*{[^}]*x_min\s*:\s*([-+]?[0-9]*\.?[0-9]+)[^}]*x_max\s*:\s*([-+]?[0-9]*\.?[0-9]+)[^}]*y_min\s*:\s*([-+]?[0-9]*\.?[0-9]+)[^}]*y_max\s*:\s*([-+]?[0-9]*\.?[0-9]+)[^}]*z_min\s*:\s*([-+]?[0-9]*\.?[0-9]+)[^}]*z_max\s*:\s*([-+]?[0-9]*\.?[0-9]+)'
            
#             # Pattern 2: Format without braces
#             bounds_pattern2 = r'workspace_boundaries\s*:\s*x_min\s*:\s*([-+]?[0-9]*\.?[0-9]+)\s*x_max\s*:\s*([-+]?[0-9]*\.?[0-9]+)\s*y_min\s*:\s*([-+]?[0-9]*\.?[0-9]+)\s*y_max\s*:\s*([-+]?[0-9]*\.?[0-9]+)\s*z_min\s*:\s*([-+]?[0-9]*\.?[0-9]+)\s*z_max\s*:\s*([-+]?[0-9]*\.?[0-9]+)'
            
#             # Try individual parameter extraction if block extraction fails
#             x_min_pattern = r'x_min\s*:\s*([-+]?[0-9]*\.?[0-9]+)'
#             x_max_pattern = r'x_max\s*:\s*([-+]?[0-9]*\.?[0-9]+)'
#             y_min_pattern = r'y_min\s*:\s*([-+]?[0-9]*\.?[0-9]+)'
#             y_max_pattern = r'y_max\s*:\s*([-+]?[0-9]*\.?[0-9]+)'
#             z_min_pattern = r'z_min\s*:\s*([-+]?[0-9]*\.?[0-9]+)'
#             z_max_pattern = r'z_max\s*:\s*([-+]?[0-9]*\.?[0-9]+)'
            
#             # Try pattern 1
#             bounds_match = re.search(bounds_pattern1, content, re.DOTALL)
#             if bounds_match:
#                 result['boundaries'] = {
#                     'x_min': float(bounds_match.group(1)),
#                     'x_max': float(bounds_match.group(2)),
#                     'y_min': float(bounds_match.group(3)),
#                     'y_max': float(bounds_match.group(4)),
#                     'z_min': float(bounds_match.group(5)),
#                     'z_max': float(bounds_match.group(6))
#                 }
#             else:
#                 # Try pattern 2
#                 bounds_match = re.search(bounds_pattern2, content, re.DOTALL)
#                 if bounds_match:
#                     result['boundaries'] = {
#                         'x_min': float(bounds_match.group(1)),
#                         'x_max': float(bounds_match.group(2)),
#                         'y_min': float(bounds_match.group(3)),
#                         'y_max': float(bounds_match.group(4)),
#                         'z_min': float(bounds_match.group(5)),
#                         'z_max': float(bounds_match.group(6))
#                     }
#                 else:
#                     # Try individual parameter extraction
#                     x_min_match = re.search(x_min_pattern, content)
#                     x_max_match = re.search(x_max_pattern, content)
#                     y_min_match = re.search(y_min_pattern, content)
#                     y_max_match = re.search(y_max_pattern, content)
#                     z_min_match = re.search(z_min_pattern, content)
#                     z_max_match = re.search(z_max_pattern, content)
                    
#                     if all([x_min_match, x_max_match, y_min_match, y_max_match, z_min_match, z_max_match]):
#                         result['boundaries'] = {
#                             'x_min': float(x_min_match.group(1)),
#                             'x_max': float(x_max_match.group(1)),
#                             'y_min': float(y_min_match.group(1)),
#                             'y_max': float(y_max_match.group(1)),
#                             'z_min': float(z_min_match.group(1)),
#                             'z_max': float(z_max_match.group(1))
#                         }
#                     else:
#                         print("Warning: Could not parse workspace boundaries using individual extraction")
#                         print("Dumping relevant portion of file content for debugging:")
                        
#                         # Find the workspace_boundaries section and print it
#                         section_match = re.search(r'workspace_boundaries.*?[{}]', content, re.DOTALL)
#                         if section_match:
#                             print(section_match.group(0))
            
#             if 'boundaries' in result:
#                 print("Successfully parsed workspace boundaries:")
#                 for key, value in result['boundaries'].items():
#                     print(f"  {key}: {value:.4f}")
#             else:
#                 print("Warning: Could not parse workspace boundaries")
            
#             # Extract markers
#             markers = []
#             marker_pattern = r'id:(\d+),\s*position:\s*\[\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*\]'
            
#             for match in re.finditer(marker_pattern, content):
#                 marker_id = int(match.group(1))
#                 x = float(match.group(2))
#                 y = float(match.group(3))
#                 z = float(match.group(4))
#                 markers.append({
#                     'id': marker_id,
#                     'position': [x, y, z]
#                 })
            
#             result['markers'] = markers
            
#             print(f"Successfully parsed workspace file with {len(markers)} markers")
#             return result
        
#         # Check if this is an extrinsic matrix YAML
#         elif 'bHc' in content:
#             # Try to extract the 4x4 matrix values
#             # First, look for a data block with 16 values
#             data_pattern = r'data:\s*\[(.*?)\]'
#             data_match = re.search(data_pattern, content, re.DOTALL)
            
#             if data_match:
#                 data_text = data_match.group(1).replace('\n', ' ')
#                 values = re.findall(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', data_text)
                
#                 if len(values) == 16:  # 4x4 matrix
#                     matrix = np.array([float(v) for v in values]).reshape(4, 4)
#                     print(f"Successfully parsed extrinsic matrix:")
#                     print(matrix)
#                     return matrix
            
#             # If we couldn't parse it, use a fallback
#             print("Warning: Could not parse extrinsic matrix format, using fallback")
#             return np.array([
#                 [1.0, 0.0, 0.0, 0.3],
#                 [0.0, 0.7071, -0.7071, 0.0],
#                 [0.0, 0.7071, 0.7071, 0.5],
#                 [0.0, 0.0, 0.0, 1.0]
#             ])
        
#         # Unknown file format
#         print(f"Warning: Unknown file format in {file_path}")
#         return None
        
#     except Exception as e:
#         print(f"Error parsing file {file_path}: {e}")
#         return None

# def create_marker_vertices(center, size=0.02):
#     """Create vertices for a marker cube centered at the given point"""
#     half_size = size / 2
#     vertices = np.array([
#         [center[0] - half_size, center[1] - half_size, center[2] - half_size],  # 0
#         [center[0] + half_size, center[1] - half_size, center[2] - half_size],  # 1
#         [center[0] + half_size, center[1] + half_size, center[2] - half_size],  # 2
#         [center[0] - half_size, center[1] + half_size, center[2] - half_size],  # 3
#         [center[0] - half_size, center[1] - half_size, center[2] + half_size],  # 4
#         [center[0] + half_size, center[1] - half_size, center[2] + half_size],  # 5
#         [center[0] + half_size, center[1] + half_size, center[2] + half_size],  # 6
#         [center[0] - half_size, center[1] + half_size, center[2] + half_size]   # 7
#     ])
#     return vertices

# def create_workspace_from_boundaries(boundaries, marker_positions_robot):
#     """Create workspace vertices using the actual workspace boundaries from the YAML file"""
#     # Extract boundaries
#     x_min = boundaries['x_min']
#     x_max = boundaries['x_max']
#     y_min = boundaries['y_min']
#     y_max = boundaries['y_max']
#     z_min = boundaries['z_min']
#     z_max = boundaries['z_max']
    
#     # Create vertices of the workspace cuboid (corners in 3D space)
#     workspace_vertices = np.array([
#         # Bottom face (z = z_min)
#         [x_min, y_min, z_min],  # 0: bottom-left-front
#         [x_max, y_min, z_min],  # 1: bottom-right-front
#         [x_max, y_max, z_min],  # 2: bottom-right-back
#         [x_min, y_max, z_min],  # 3: bottom-left-back
        
#         # Top face (z = z_max)
#         [x_min, y_min, z_max],  # 4: top-left-front
#         [x_max, y_min, z_max],  # 5: top-right-front
#         [x_max, y_max, z_max],  # 6: top-right-back
#         [x_min, y_max, z_max],  # 7: top-left-back
#     ])
    
#     print(f"Created workspace from actual boundaries:")
#     print(f"  X: [{x_min:.4f}, {x_max:.4f}] (width: {x_max-x_min:.4f}m)")
#     print(f"  Y: [{y_min:.4f}, {y_max:.4f}] (depth: {y_max-y_min:.4f}m)")
#     print(f"  Z: [{z_min:.4f}, {z_max:.4f}] (height: {z_max-z_min:.4f}m)")
    
#     # Sort markers to form a proper rectangle (using convex hull approach)
#     markers_xy = marker_positions_robot[:, :2]  # Just use X,Y coordinates
#     center = np.mean(markers_xy, axis=0)
    
#     # Sort markers by angle around center
#     angles = np.arctan2(markers_xy[:, 1] - center[1], markers_xy[:, 0] - center[0])
#     sorted_indices = np.argsort(angles)
    
#     return sorted_indices, workspace_vertices

# def plot_workspace_robot_coords():
#     """Plot the workspace in robot coordinates, reading data from files"""
#     # Check if files exist
#     if not os.path.exists(WORKSPACE_YAML):
#         print(f"Error: Workspace file {WORKSPACE_YAML} not found")
#         return
    
#     if not os.path.exists(EXTRINSIC_YAML):
#         print(f"Error: Extrinsic file {EXTRINSIC_YAML} not found")
#         return
    
#     # Parse the files
#     workspace_data = parse_opencv_yaml(WORKSPACE_YAML)
#     extrinsic_matrix = parse_opencv_yaml(EXTRINSIC_YAML)
    
#     if workspace_data is None or extrinsic_matrix is None:
#         print("Error: Failed to parse one or both files")
#         return
    
#     # Get camera position in robot coordinates (from extrinsic matrix)
#     camera_pos_robot = extrinsic_matrix[:3, 3]
#     print(f"Camera position in robot coordinates: {camera_pos_robot}")
    
#     # Extract marker data in robot coordinates
#     markers = workspace_data['markers']
#     marker_positions_robot = np.array([marker['position'] for marker in markers])
#     marker_ids = [marker['id'] for marker in markers]
    
#     # Print marker positions in robot coordinates
#     print("\nMarker positions in robot coordinates:")
#     for i, (pos, id) in enumerate(zip(marker_positions_robot, marker_ids)):
#         print(f"  Marker {id}: {pos}")
    
#     # Create workspace vertices using boundaries from the YAML file
#     if 'boundaries' in workspace_data:
#         sorted_indices, workspace_vertices = create_workspace_from_boundaries(
#             workspace_data['boundaries'], marker_positions_robot)
#     else:
#         # Fallback: Calculate workspace from markers if we couldn't parse boundaries
#         print("Using fallback: Calculating workspace boundaries from markers...")
        
#         # Find the X/Y extents from markers
#         x_min = np.min(marker_positions_robot[:, 0]) - 0.01  # 1cm margin
#         x_max = np.max(marker_positions_robot[:, 0]) + 0.01
#         y_min = np.min(marker_positions_robot[:, 1]) - 0.01
#         y_max = np.max(marker_positions_robot[:, 1]) + 0.01
        
#         # For Z, use the marker height as z_min
#         z_min = np.min(marker_positions_robot[:, 2]) - 0.01
        
#         # For z_max, use the camera position minus 3cm (default offset)
#         camera_offset = 0.03  # 3cm from camera (default from C++ code)
#         z_max = camera_pos_robot[2] - camera_offset + 0.01
        
#         # Safety check to ensure z_min < z_max
#         if z_min > z_max:
#             z_min, z_max = z_max, z_min
            
#         # Create a boundaries dict
#         boundaries = {
#             'x_min': x_min,
#             'x_max': x_max,
#             'y_min': y_min,
#             'y_max': y_max,
#             'z_min': z_min,
#             'z_max': z_max
#         }
        
#         workspace_data['boundaries'] = boundaries
#         sorted_indices, workspace_vertices = create_workspace_from_boundaries(
#             boundaries, marker_positions_robot)
        
#         print("Calculated workspace boundaries:")
#         for key, value in boundaries.items():
#             print(f"  {key}: {value:.4f}")
    
#     # Create reordered marker positions and IDs
#     sorted_markers = marker_positions_robot[sorted_indices]
#     sorted_ids = [marker_ids[i] for i in sorted_indices]
    
#     # Extract vertices for different parts of the workspace
#     bottom_vertices = workspace_vertices[:4]  # First 4 are the bottom vertices (table level)
#     top_vertices = workspace_vertices[4:8]  # Last 4 are the top vertices (max height)
    
#     # Calculate overall min/max values for all elements
#     all_points = np.vstack([
#         workspace_vertices,
#         marker_positions_robot,
#         camera_pos_robot.reshape(1, 3),  # Camera position
#         np.array([[0, 0, 0]])  # Robot origin
#     ])
    
#     min_x, max_x = np.min(all_points[:, 0]) - 0.05, np.max(all_points[:, 0]) + 0.05
#     min_y, max_y = np.min(all_points[:, 1]) - 0.05, np.max(all_points[:, 1]) + 0.05
#     min_z, max_z = np.min(all_points[:, 2]) - 0.05, np.max(all_points[:, 2]) + 0.05
    
#     print(f"\nOverall coordinate ranges (with margin):")
#     print(f"  X: [{min_x:.4f}, {max_x:.4f}]")
#     print(f"  Y: [{min_y:.4f}, {max_y:.4f}]")
#     print(f"  Z: [{min_z:.4f}, {max_z:.4f}]")
    
#     # Set up the figure and axis
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Define the faces of the workspace cuboid
#     faces = [
#         # Bottom face
#         [bottom_vertices[0], bottom_vertices[1], bottom_vertices[2], bottom_vertices[3]],
#         # Top face
#         [top_vertices[0], top_vertices[1], top_vertices[2], top_vertices[3]],
#         # Front face
#         [bottom_vertices[0], bottom_vertices[1], top_vertices[1], top_vertices[0]],
#         # Right face
#         [bottom_vertices[1], bottom_vertices[2], top_vertices[2], top_vertices[1]],
#         # Back face
#         [bottom_vertices[2], bottom_vertices[3], top_vertices[3], top_vertices[2]],
#         # Left face
#         [bottom_vertices[3], bottom_vertices[0], top_vertices[0], top_vertices[3]]
#     ]
    
#     # Create the workspace with all faces
#     workspace = Poly3DCollection(faces, alpha=0.35, linewidths=1.5, edgecolor='darkblue')
#     workspace.set_facecolor('cyan')
#     ax.add_collection3d(workspace)
    
#     # Draw connecting lines to make the structure clearer
#     # Top edges
#     ax.plot([top_vertices[0, 0], top_vertices[1, 0]], 
#             [top_vertices[0, 1], top_vertices[1, 1]], 
#             [top_vertices[0, 2], top_vertices[1, 2]], 'b-', alpha=0.7, linewidth=1.5)
#     ax.plot([top_vertices[1, 0], top_vertices[2, 0]], 
#             [top_vertices[1, 1], top_vertices[2, 1]], 
#             [top_vertices[1, 2], top_vertices[2, 2]], 'b-', alpha=0.7, linewidth=1.5)
#     ax.plot([top_vertices[2, 0], top_vertices[3, 0]], 
#             [top_vertices[2, 1], top_vertices[3, 1]], 
#             [top_vertices[2, 2], top_vertices[3, 2]], 'b-', alpha=0.7, linewidth=1.5)
#     ax.plot([top_vertices[3, 0], top_vertices[0, 0]], 
#             [top_vertices[3, 1], top_vertices[0, 1]], 
#             [top_vertices[3, 2], top_vertices[0, 2]], 'b-', alpha=0.7, linewidth=1.5)
    
#     # Bottom edges
#     ax.plot([bottom_vertices[0, 0], bottom_vertices[1, 0]], 
#             [bottom_vertices[0, 1], bottom_vertices[1, 1]], 
#             [bottom_vertices[0, 2], bottom_vertices[1, 2]], 'b-', alpha=0.7, linewidth=1.5)
#     ax.plot([bottom_vertices[1, 0], bottom_vertices[2, 0]], 
#             [bottom_vertices[1, 1], bottom_vertices[2, 1]], 
#             [bottom_vertices[1, 2], bottom_vertices[2, 2]], 'b-', alpha=0.7, linewidth=1.5)
#     ax.plot([bottom_vertices[2, 0], bottom_vertices[3, 0]], 
#             [bottom_vertices[2, 1], bottom_vertices[3, 1]], 
#             [bottom_vertices[2, 2], bottom_vertices[3, 2]], 'b-', alpha=0.7, linewidth=1.5)
#     ax.plot([bottom_vertices[3, 0], bottom_vertices[0, 0]], 
#             [bottom_vertices[3, 1], bottom_vertices[0, 1]], 
#             [bottom_vertices[3, 2], bottom_vertices[0, 2]], 'b-', alpha=0.7, linewidth=1.5)
    
#     # Vertical edges
#     for i in range(4):
#         ax.plot([bottom_vertices[i, 0], top_vertices[i, 0]], 
#                 [bottom_vertices[i, 1], top_vertices[i, 1]], 
#                 [bottom_vertices[i, 2], top_vertices[i, 2]], 'b-', alpha=0.7, linewidth=1.5)
    
#     # ----- Plot the markers as cubes -----
#     marker_colors = ['orange', 'gold', 'darkorange', 'orangered']
    
#     for i, (pos, id) in enumerate(zip(marker_positions_robot, marker_ids)):
#         # Create small cube for each marker
#         marker_vertices = create_marker_vertices(pos, size=0.02)
        
#         # Define the faces of the cube
#         marker_faces = [
#             [marker_vertices[0], marker_vertices[1], marker_vertices[2], marker_vertices[3]],  # Bottom
#             [marker_vertices[4], marker_vertices[5], marker_vertices[6], marker_vertices[7]],  # Top
#             [marker_vertices[0], marker_vertices[1], marker_vertices[5], marker_vertices[4]],  # Front
#             [marker_vertices[2], marker_vertices[3], marker_vertices[7], marker_vertices[6]],  # Back
#             [marker_vertices[0], marker_vertices[3], marker_vertices[7], marker_vertices[4]],  # Left
#             [marker_vertices[1], marker_vertices[2], marker_vertices[6], marker_vertices[5]]   # Right
#         ]
        
#         # Create marker cube
#         marker_cube = Poly3DCollection(marker_faces, alpha=0.7, linewidths=1, edgecolor='k')
#         marker_cube.set_facecolor(marker_colors[i % len(marker_colors)])
#         ax.add_collection3d(marker_cube)
        
#         # Add marker label
#         ax.text(pos[0], pos[1], pos[2], f"  Marker {id}", fontsize=10)
        
#     # ----- Plot the marker table (connecting the markers) -----
#     # Use sorted markers to draw the table outline
#     table_points = np.vstack([sorted_markers, sorted_markers[0]])  # Close the loop
    
#     # Draw the table outline
#     ax.plot(table_points[:, 0], table_points[:, 1], table_points[:, 2], 'r-', linewidth=2, label="Marker Table")
    
#     # ----- Plot camera position in robot coordinates -----
#     ax.scatter(camera_pos_robot[0], camera_pos_robot[1], camera_pos_robot[2], 
#               c='purple', marker='^', s=200, label='Camera')
#     ax.text(camera_pos_robot[0], camera_pos_robot[1], camera_pos_robot[2], 
#            "  Camera", fontsize=12)
    
#     # ----- Draw an outline of camera viewing direction -----
#     # Create a line from camera toward the center of markers
#     marker_center = np.mean(marker_positions_robot, axis=0)
#     direction = marker_center - camera_pos_robot
#     direction = direction / np.linalg.norm(direction) * 0.2  # Normalize and scale
    
#     # Draw camera viewing direction
#     ax.plot([camera_pos_robot[0], camera_pos_robot[0] + direction[0]],
#             [camera_pos_robot[1], camera_pos_robot[1] + direction[1]],
#             [camera_pos_robot[2], camera_pos_robot[2] + direction[2]],
#             'purple', linestyle='--', linewidth=2, alpha=0.7)
    
#     # ----- Draw robot coordinate system axes at origin -----
#     origin = np.array([0, 0, 0])
#     axis_length = 0.15
    
#     # X-axis (red)
#     ax.plot([origin[0], origin[0] + axis_length], 
#             [origin[1], origin[1]], 
#             [origin[2], origin[2]], 'r-', linewidth=2)
#     ax.text(origin[0] + axis_length*1.1, origin[1], origin[2], "X", color='red', fontsize=12)
    
#     # Y-axis (green)
#     ax.plot([origin[0], origin[0]], 
#             [origin[1], origin[1] + axis_length], 
#             [origin[2], origin[2]], 'g-', linewidth=2)
#     ax.text(origin[0], origin[1] + axis_length*1.1, origin[2], "Y", color='green', fontsize=12)
    
#     # Z-axis (blue)
#     ax.plot([origin[0], origin[0]], 
#             [origin[1], origin[1]], 
#             [origin[2], origin[2] + axis_length], 'b-', linewidth=2)
#     ax.text(origin[0], origin[1], origin[2] + axis_length*1.1, "Z", color='blue', fontsize=12)
    
#     # Add a text label for the robot origin
#     ax.text(origin[0], origin[1], origin[2], "  Robot Origin", fontsize=10)
    
#     # ----- Set labels and title -----
#     ax.set_xlabel('X (m)')
#     ax.set_ylabel('Y (m)')
#     ax.set_zlabel('Z (m)')
#     ax.set_title('Workspace in Robot Coordinates (Using Actual Boundaries)')
    
#     # ----- Create legend -----
#     cuboid_patch = mpatches.Patch(color='cyan', alpha=0.35, label='Workspace Volume')
#     marker_patch = mpatches.Patch(color='orange', alpha=0.7, label='ArUco Markers')
#     table_line = mlines.Line2D([], [], color='red', linewidth=2, label='Marker Table')
#     purple_triangle = mlines.Line2D([], [], color='purple', marker='^', linestyle='None',
#                                   markersize=10, label='Camera')
#     red_line = mlines.Line2D([], [], color='red', linewidth=2, label='X-axis')
#     green_line = mlines.Line2D([], [], color='green', linewidth=2, label='Y-axis')
#     blue_line = mlines.Line2D([], [], color='blue', linewidth=2, label='Z-axis')
    
#     # Add legend
#     ax.legend(handles=[cuboid_patch, marker_patch, table_line, purple_triangle, 
#                        red_line, green_line, blue_line],
#              loc='upper right')
    
#     # ----- Set axis limits -----
#     # Use the calculated bounds
#     ax.set_xlim(min_x, max_x)
#     ax.set_ylim(min_y, max_y)
#     ax.set_zlim(min_z, max_z)
    
#     # ----- Set initial view -----
#     # Try a different view angle to see the full workspace
#     ax.view_init(elev=30, azim=120)
    
#     # ----- Display note -----
#     boundaries = workspace_data['boundaries']
#     note_text = (f"Note: Workspace boundaries in robot coordinates are:\n"
#                 f"X: [{boundaries['x_min']:.3f}, {boundaries['x_max']:.3f}], "
#                 f"Y: [{boundaries['y_min']:.3f}, {boundaries['y_max']:.3f}], "
#                 f"Z: [{boundaries['z_min']:.3f}, {boundaries['z_max']:.3f}]")
#     fig.text(0.5, 0.01, note_text, ha='center', fontsize=12)
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     plot_workspace_robot_coords()