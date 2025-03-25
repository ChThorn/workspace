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
OUTPUT_FILE = None  # Set to None to show interactive plot instead of saving

def parse_opencv_matrix(content):
    """Parse OpenCV matrix data from YAML content"""
    # Extract data from the matrix
    rows_match = re.search(r'rows:\s*(\d+)', content)
    cols_match = re.search(r'cols:\s*(\d+)', content)
    data_match = re.search(r'data:\s*\[(.*?)\]', content, re.DOTALL)
    
    if not (rows_match and cols_match and data_match):
        return None  # Failed to parse matrix
    
    try:
        rows = int(rows_match.group(1))
        cols = int(cols_match.group(1))
        data_str = data_match.group(1).replace('\n', ' ').strip()
        
        # Parse data values
        data_values = []
        current_value = ""
        in_quotes = False
        
        for char in data_str:
            if char == '"' or char == "'":
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                if current_value.strip():
                    data_values.append(float(current_value.strip()))
                current_value = ""
            else:
                current_value += char
        
        # Add the last value if it exists
        if current_value.strip():
            data_values.append(float(current_value.strip()))
        
        return {'rows': rows, 'cols': cols, 'data': data_values}
    except Exception as e:
        print(f"Error parsing matrix: {e}")
        return None

def parse_opencv_yaml(file_path):
    """Parse an OpenCV YAML file manually"""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Remove the YAML directive
        content = re.sub(r'%YAML:[\d\.]+', '', content)
        content = re.sub(r'%YAML[\s-]*[\d\.]+', '', content)
        
        # Handle workspace_boundaries format
        if 'workspace_boundaries' in content:
            try:
                # Extract workspace boundaries
                bounds_match = re.search(r'workspace_boundaries:(.*?)markers:', content, re.DOTALL)
                if bounds_match:
                    bounds_str = bounds_match.group(1)
                    
                    # Extract each value
                    x_min = float(re.search(r'x_min:\s*([-\d\.e]+)', bounds_str).group(1))
                    x_max = float(re.search(r'x_max:\s*([-\d\.e]+)', bounds_str).group(1))
                    y_min = float(re.search(r'y_min:\s*([-\d\.e]+)', bounds_str).group(1))
                    y_max = float(re.search(r'y_max:\s*([-\d\.e]+)', bounds_str).group(1))
                    z_min = float(re.search(r'z_min:\s*([-\d\.e]+)', bounds_str).group(1))
                    z_max = float(re.search(r'z_max:\s*([-\d\.e]+)', bounds_str).group(1))
                    
                    workspace_boundaries = {
                        'x_min': x_min, 'x_max': x_max,
                        'y_min': y_min, 'y_max': y_max,
                        'z_min': z_min, 'z_max': z_max
                    }
                else:
                    workspace_boundaries = {}
                
                # Extract markers
                markers = []
                marker_matches = re.finditer(r'id:(\d+),\s*position:\s*\[\s*([-\d\.e]+),\s*([-\d\.e]+),\s*([-\d\.e]+)\s*\]', content)
                for match in marker_matches:
                    marker_id = int(match.group(1))
                    x = float(match.group(2))
                    y = float(match.group(3))
                    z = float(match.group(4))
                    markers.append({'id': marker_id, 'position': [x, y, z]})
                
                return {'workspace_boundaries': workspace_boundaries, 'markers': markers}
            except Exception as e:
                print(f"Error parsing workspace file: {e}")
                return None
        
        # Handle extrinsic matrix format
        elif 'bHc' in content:
            try:
                # Find the bHc matrix section
                bhc_match = re.search(r'bHc:(.*?)(?:\Z|[a-zA-Z])', content, re.DOTALL)
                if not bhc_match:
                    return None
                
                bhc_content = bhc_match.group(1)
                
                # Check if it contains an opencv-matrix
                if '!!opencv-matrix' in bhc_content:
                    matrix = parse_opencv_matrix(bhc_content)
                    if matrix:
                        return {'bHc': matrix}
                
                return None
            except Exception as e:
                print(f"Error parsing extrinsic file: {e}")
                return None
        
        return None
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        print(f"Make sure {file_path} exists or update the file paths in the script.")
        return None
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        print("Check if the file exists and is a valid YAML file.")
        return None

def plot_workspace():
    """Plot the 3D workspace based on the YAML files"""
    print(f"Loading workspace data from: {WORKSPACE_YAML}")
    print(f"Loading extrinsic data from: {EXTRINSIC_YAML}")

    # Check if files exist
    if not os.path.exists(WORKSPACE_YAML):
        print(f"Error: {WORKSPACE_YAML} not found. Please check the path.")
        exit(1)
    
    # Load data from YAML files
    try:
        data = parse_opencv_yaml(WORKSPACE_YAML)
        if not data:
            print(f"Failed to parse workspace file: {WORKSPACE_YAML}")
            exit(1)
        
        # Try to load extrinsic, but proceed even if not available
        camera_pos = None
        try:
            if os.path.exists(EXTRINSIC_YAML):
                extrinsic_data = parse_opencv_yaml(EXTRINSIC_YAML)
                if extrinsic_data and 'bHc' in extrinsic_data:
                    # Reshape the matrix
                    matrix_data = extrinsic_data['bHc']['data']
                    rows = extrinsic_data['bHc']['rows']
                    cols = extrinsic_data['bHc']['cols']
                    camera_matrix = np.array(matrix_data).reshape(rows, cols)
                    camera_pos = camera_matrix[:3, 3]
                    print(f"Using camera position from extrinsic file: {camera_pos}")
            
            if camera_pos is None:
                # Default camera position if extrinsic file is not available or couldn't be parsed
                camera_pos = np.array([0.3, 0, 0.5])
                print(f"Using default camera position: {camera_pos}")
        except Exception as e:
            # Default camera position if there's any error with extrinsic file
            camera_pos = np.array([0.3, 0, 0.5])
            print(f"Error processing extrinsic file: {e}")
            print(f"Using default camera position: {camera_pos}")
    except Exception as e:
        print(f"Error loading YAML files: {e}")
        exit(1)
    
    # Extract workspace boundaries
    boundaries = data['workspace_boundaries']
    x_min, x_max = boundaries['x_min'], boundaries['x_max']
    y_min, y_max = boundaries['y_min'], boundaries['y_max']
    z_min, z_max = boundaries['z_min'], boundaries['z_max']
    
    print(f"Workspace boundaries:")
    print(f"  X: [{x_min:.4f}, {x_max:.4f}] - width: {x_max-x_min:.4f}m")
    print(f"  Y: [{y_min:.4f}, {y_max:.4f}] - length: {y_max-y_min:.4f}m")
    print(f"  Z: [{z_min:.4f}, {z_max:.4f}] - height: {z_max-z_min:.4f}m")
    
    # Extract marker positions
    markers = data['markers']
    marker_positions = []
    marker_ids = []
    for marker in markers:
        marker_ids.append(marker['id'])
        marker_positions.append(marker['position'])
    marker_positions = np.array(marker_positions)
    
    # Calculate average Z position for table level
    table_z = np.mean(marker_positions[:, 2])
    print(f"Table level (average marker Z): {table_z:.4f}m")
    
    # Set up the figure and axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the workspace cuboid
    # Create the 8 vertices of the cuboid
    vertices = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
    ])
    
    # Define the faces of the cuboid using the vertices
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left face
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # Right face
    ]
    
    # Create the cuboid
    cuboid = Poly3DCollection(faces, alpha=0.2, linewidths=1, edgecolor='k')
    cuboid.set_facecolor('cyan')
    ax.add_collection3d(cuboid)
    
    # Draw the table level (red rectangle)
    table_corners = np.array([
        [x_min, y_min, table_z], [x_max, y_min, table_z],
        [x_max, y_max, table_z], [x_min, y_max, table_z],
        [x_min, y_min, table_z]  # Repeat first point to close the loop
    ])
    ax.plot(table_corners[:, 0], table_corners[:, 1], table_corners[:, 2], 'r-', linewidth=2)
    
    # Draw the top level (blue rectangle)
    top_z = z_max  # Camera Z - offset
    top_corners = np.array([
        [x_min, y_min, top_z], [x_max, y_min, top_z],
        [x_max, y_max, top_z], [x_min, y_max, top_z],
        [x_min, y_min, top_z]  # Repeat first point to close the loop
    ])
    ax.plot(top_corners[:, 0], top_corners[:, 1], top_corners[:, 2], 'b-', linewidth=2)
    
    # Connect the table and top levels with green lines
    for i in range(4):
        ax.plot([table_corners[i, 0], top_corners[i, 0]],
                [table_corners[i, 1], top_corners[i, 1]],
                [table_corners[i, 2], top_corners[i, 2]], 'g-', linewidth=2)
    
    # Plot the marker positions
    ax.scatter(marker_positions[:, 0], marker_positions[:, 1], marker_positions[:, 2], 
               c='orange', marker='o', s=100, label='Markers')
    
    # Add marker labels
    for i, (pos, id) in enumerate(zip(marker_positions, marker_ids)):
        ax.text(pos[0], pos[1], pos[2], f"  Marker {id}", fontsize=10)
    
    # Plot camera position
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], c='purple', marker='^', s=200, label='Camera')
    ax.text(camera_pos[0], camera_pos[1], camera_pos[2], "  Camera", fontsize=12)
    
    # Draw a simple camera frustum to show field of view
    # Direction vector from camera to center of markers
    target = np.mean(marker_positions, axis=0)
    direction = target - camera_pos
    dir_vec = direction / np.linalg.norm(direction)
    
    # Calculate perpendicular vectors to create frustum corners
    perp1 = np.cross(dir_vec, np.array([0, 0, 1]))
    if np.linalg.norm(perp1) < 1e-5:  # If dir_vec is aligned with Z
        perp1 = np.array([1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1) * 0.05
    perp2 = np.cross(dir_vec, perp1)
    perp2 = perp2 / np.linalg.norm(perp2) * 0.05
    
    # Create frustum corners near camera
    frustum_corners = [
        camera_pos + perp1 + perp2,
        camera_pos + perp1 - perp2,
        camera_pos - perp1 - perp2,
        camera_pos - perp1 + perp2,
        camera_pos + perp1 + perp2  # Repeat first point to close the loop
    ]
    
    # Plot frustum outline
    ax.plot([p[0] for p in frustum_corners], 
            [p[1] for p in frustum_corners], 
            [p[2] for p in frustum_corners], 
            color='purple', alpha=0.3, linewidth=1)
    
    # Draw lines from camera to frustum corners
    for i in range(4):
        ax.plot([camera_pos[0], frustum_corners[i][0]], 
                [camera_pos[1], frustum_corners[i][1]], 
                [camera_pos[2], frustum_corners[i][2]], 
                color='purple', alpha=0.3, linewidth=1)
    
    # Draw lines from frustum to markers for visualization
    for i in range(4):
        for marker_pos in marker_positions:
            ax.plot([frustum_corners[i][0], marker_pos[0]], 
                    [frustum_corners[i][1], marker_pos[1]], 
                    [frustum_corners[i][2], marker_pos[2]], 
                    'purple', alpha=0.1, linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Workspace Visualization')
    
    # Create legend handles
    cuboid_patch = mpatches.Patch(color='cyan', alpha=0.2, label='Workspace Volume')
    red_line = mlines.Line2D([], [], color='red', linewidth=2, label='Table Level')
    blue_line = mlines.Line2D([], [], color='blue', linewidth=2, label='Top Level (3cm from Camera)')
    green_line = mlines.Line2D([], [], color='green', linewidth=2, label='Vertical Edges')
    orange_dot = mlines.Line2D([], [], color='orange', marker='o', linestyle='None', 
                              markersize=10, label='ArUco Markers')
    purple_triangle = mlines.Line2D([], [], color='purple', marker='^', linestyle='None',
                                  markersize=10, label='Camera')
    
    # Add legend
    ax.legend(handles=[cuboid_patch, red_line, blue_line, green_line, orange_dot, purple_triangle],
             loc='upper right')
    
    # Set axis limits slightly larger than the workspace
    margin = 0.1
    ax.set_xlim(min(x_min, camera_pos[0]) - margin, max(x_max, camera_pos[0]) + margin)
    ax.set_ylim(min(y_min, camera_pos[1]) - margin, max(y_max, camera_pos[1]) + margin)
    ax.set_zlim(min(z_min, camera_pos[2]) - margin, max(z_max, camera_pos[2]) + margin)
    
    # Add dimensions text
    dimensions_text = (f"Workspace Dimensions: "
                      f"{round(x_max-x_min, 2)}m × "
                      f"{round(y_max-y_min, 2)}m × "
                      f"{round(z_max-z_min, 2)}m")
    fig.text(0.5, 0.01, dimensions_text, ha='center', fontsize=12)
    
    # Add a view that shows the workspace clearly
    ax.view_init(elev=30, azim=-60)
    
    plt.tight_layout()
    
    # Save or show the plot
    if OUTPUT_FILE:
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {OUTPUT_FILE}")
    else:
        # Show interactive plot
        print("Displaying interactive 3D plot...")
        print("Tip: You can rotate the view by clicking and dragging!")
        plt.show()

if __name__ == "__main__":
    plot_workspace()