from plyfile import PlyData, PlyElement
import numpy as np

COLOR_CODING_REF = 1.0

def find_origin_and_opposite_corner(vertices):
    """
    Find the origin as the point with the minimum x, y, z values,
    and the opposite corner as the point with the maximum x, y, z values.
    """
    positions = np.array([[v['x'], v['y'], v['z']] for v in vertices])
    origin = positions.min(axis=0)
    opposite_corner = positions.max(axis=0)
    return origin, opposite_corner

def compute_color_from_distance_axis(position, origin, opposite_corner):
    """
    Compute color based on normalized distance from the specified origin.
    Color ranges from -1 to 1 at the origin to the farthest corner.
    """
    distance_x = abs(position[0] - origin[0])
    distance_y = abs(position[1] - origin[1])
    distance_z = abs(position[2] - origin[2])

    # Normalize distances between 0 and 1
    intensity_x = distance_x / (opposite_corner[0] - origin[0])
    intensity_y = distance_y / (opposite_corner[1] - origin[1])
    intensity_z = distance_z / (opposite_corner[2] - origin[2])

    #Map intensities to range -1 to 1
    color = (
        (intensity_x * 2 - 1) * COLOR_CODING_REF,
        (intensity_y * 2 - 1) * COLOR_CODING_REF,
        (intensity_z * 2 - 1) * COLOR_CODING_REF
    )

    #Mappa le intensità nell'intervallo 0-255
    #color = (
    #    int(intensity_x * 255),
    #    int(intensity_y * 255),
    #    int(intensity_z * 255)
    #)
    
    return color

def update_gaussian_colors_ply(input_ply, output_ply):
    # Read the PLY file
    ply_data = PlyData.read(input_ply)
    vertices = ply_data['vertex']
    
    # Find origin and opposite corner
    origin, opposite_corner = find_origin_and_opposite_corner(vertices)

    # Modify colors based on distance from origin
    for vertex in vertices:
        position = np.array([vertex['x'], vertex['y'], vertex['z']])
        new_color = compute_color_from_distance_axis(position, origin, opposite_corner)
        
        print(f"Position: {position}, New Color: {new_color}")
    
        # Update the vertex colors in-place with new float color values between -1 and 1
        vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2'] = new_color

    # Write the modified data back to a new PLY file
    ply_data.write(output_ply)

# Usage
input_ply = 'exports/splat/splat_filtered.ply'  # Replace with your actual input PLY file path
output_ply = 'exports/splat/splat_filtered_nocs.ply'  # Define output file path
update_gaussian_colors_ply(input_ply, output_ply)