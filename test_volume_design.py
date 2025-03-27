import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_prism(bottom_3d_points, h):
    """
    Plots a 3D prism from four 3D points by extruding them along the z-axis by height h.
    
    Parameters:
    - bottom_3d_points: List of 4 points, each as [x, y, z].
    - h: Height to extrude the shape in the z-direction.
    """
    # Check that we have exactly 4 points
    if len(bottom_3d_points) != 4:
        raise ValueError("Must provide exactly four 3D points")
    
    # Create top points by adding height h to the z-coordinate
    top_points = [[p[0], p[1], p[2] + h] for p in bottom_3d_points]
    
    # Set up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot bottom face (connect points and close the loop)
    x_bottom = [p[0] for p in bottom_3d_points] + [bottom_3d_points[0][0]]
    y_bottom = [p[1] for p in bottom_3d_points] + [bottom_3d_points[0][1]]
    z_bottom = [p[2] for p in bottom_3d_points] + [bottom_3d_points[0][2]]
    ax.plot(x_bottom, y_bottom, z_bottom, 'b-', label='Bottom face')
    
    # Plot top face (connect points and close the loop)
    x_top = [p[0] for p in top_points] + [top_points[0][0]]
    y_top = [p[1] for p in top_points] + [top_points[0][1]]
    z_top = [p[2] for p in top_points] + [top_points[0][2]]
    ax.plot(x_top, y_top, z_top, 'r-', label='Top face')
    
    # Plot sides (lines between corresponding bottom and top points)
    for i in range(4):
        ax.plot([bottom_3d_points[i][0], top_points[i][0]],
                [bottom_3d_points[i][1], top_points[i][1]],
                [bottom_3d_points[i][2], top_points[i][2]], 'g-', label='Sides' if i == 0 else "")
    
    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Optional: Set equal aspect ratio for better visualization
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        # Skip if using an older Matplotlib version
        pass
    
    # Show the plot
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define four 3D points with varying z-coordinates
    bottom_3d_points = [[0, 0, 0], [2, 0, 1], [2, 1, 0.5], [0, 1, 1.5]]
    # Set extrusion height
    h = 2
    # Plot the 3D prism
    plot_3d_prism(bottom_3d_points, h)