import numpy as np
import pandas as pd
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Polygon, box
import geopandas as gpd
from scipy.interpolate import Rbf


# 1. Functions for bone alignment
# Find the outline of the bone in the x-y plane
def find_outline(points, window_size=10):
    """
    Find the outline of a set of points by finding the min and max y-values for each x-value within a window.
    The outline is only in the x-y plane.

    Parameters:
    points : np.array of shape (n,2) points.
    window_size : size of the window to smooth the outline.
    Returns:
    outline_points : np.array of outline points.
    """
    df = pd.DataFrame(points, columns=["x", "y"])

    min_y_points = []
    max_y_points = []

    # Sort points by x value
    df_sorted = df.sort_values(by="x")

    # Slide over the x values with a window
    for i in range(0, len(df_sorted), window_size):
        window = df_sorted.iloc[i:i + window_size]
        min_y = window.loc[window["y"].idxmin()]
        max_y = window.loc[window["y"].idxmax()]
        min_y_points.append(min_y)
        max_y_points.append(max_y)

    # Ensure the outline is in order
    min_y_points = pd.DataFrame(
        min_y_points).drop_duplicates().sort_values(by="x").values
    max_y_points = pd.DataFrame(max_y_points).drop_duplicates(
    ).sort_values(by="x", ascending=False).values

    # Combine min_y and max_y points and close the loop
    outline_points = np.vstack([min_y_points, max_y_points, min_y_points[0]])

    return outline_points

# Find the center of each bone (outline) and put the center of the bones at the same position


def calculate_centroid(outline_points):
    """
    Calculate the centroid of the bone outline.

    Parameters:
    outline_points : np.array of shape (n, 2)

    Returns:
    centroid : tuple containing (centroid_x, centroid_y)
    """
    # Use the weight centroid
    centroid_x = np.mean(outline_points[:, 0])
    centroid_y = np.mean(outline_points[:, 1])

    # Using the middle of the x and y values as the centroid
    # x_min, x_max = outline_points[:, 0].min(), outline_points[:, 0].max()
    # y_min, y_max = outline_points[:, 1].min(), outline_points[:, 1].max()
    # centroid_x = (x_min + x_max) / 2
    # centroid_y = (y_min + y_max) / 2

    return centroid_x, centroid_y


def translate_to_origin(outline_points, centroid):
    """
    Translate the outline points so that the centroid is at the origin.

    Parameters:
    outline_points : np.array of shape (n, 2)
    centroid : tuple containing (centroid_x, centroid_y)

    Returns:
    translated_points : np.array of shape (n, 2)
    """
    translated_points = outline_points.astype(np.float64).copy()

    translated_points[:, 0] -= centroid[0]
    translated_points[:, 1] -= centroid[1]
    return translated_points

# Rescale the bones to the same size (bounding box)(optional)


def get_max_dimensions(bone_dicts):
    """
    Find the maximum width and height across all bone outlines in the given dictionaries.

    Parameters:
    bone_dicts : list of dictionaries of bones (where each value is a DataFrame with "Position.X" and "Position.Y")

    Returns:
    max_width : float, maximum width found across all bones
    max_height : float, maximum height found across all bones
    """
    max_width = 0
    max_height = 0

    for bone_dict in bone_dicts:
        for df in bone_dict.values():
            # Extract bone points where "source" == "Bone"
            bone_points = df[df["source"] == "Bone"][[
                "Position.X", "Position.Y"]].values

            # Find min and max values of x and y
            min_x, max_x = bone_points[:, 0].min(), bone_points[:, 0].max()
            min_y, max_y = bone_points[:, 1].min(), bone_points[:, 1].max()

            # Calculate width and height
            width = max_x - min_x
            height = max_y - min_y

            # Update maximum width and height if necessary
            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height

    return max_width, max_height


def rescale_outline(outline_points, max_width, max_height):
    """
    Rescale the bone outline to fit within the maximum width and height across all bones.

    Parameters:
    outline_points : np.array of shape (n, 2)
    max_width : float, the maximum width across all bones
    max_height : float, the maximum height across all bones

    Returns:
    scaled_points : np.array of shape (n, 2)
    """
    min_x, max_x = outline_points[:, 0].min(), outline_points[:, 0].max()
    min_y, max_y = outline_points[:, 1].min(), outline_points[:, 1].max()

    current_width = max_x - min_x
    current_height = max_y - min_y

    scale_x = max_width / current_width
    scale_y = max_height / current_height

    scaled_points = outline_points.copy()
    scaled_points[:, 0] *= scale_x
    scaled_points[:, 1] *= scale_y

    return scaled_points

# Calculate the overlap area on the grid inside the outline


def calculate_overlap_area(reference_outline, target_outline, resolution=200):
    """
    Calculate the overlap area (in terms of pixels or points) between two outlines.

    Parameters:
    reference_outline : np.array of shape (n, 2), outline of the reference bone
    target_outline : np.array of shape (n, 2), outline of the target bone
    resolution : int, the number of points or pixels to use for the area calculation.

    Returns:
    overlap_area : float, the number of pixels or points where the areas overlap.
    """
    # Ensure the outlines are 2D arrays of shape (N, 2)
    reference_outline = np.asarray(reference_outline).reshape(-1, 2)
    target_outline = np.asarray(target_outline).reshape(-1, 2)

    # Calculate centroids for both bones
    reference_centroid = calculate_centroid(reference_outline)
    target_centroid = calculate_centroid(target_outline)

    # Translate both bones to center them
    reference_outline_centered = translate_to_origin(
        reference_outline, reference_centroid)
    target_outline_centered = translate_to_origin(
        target_outline, target_centroid)

    # Get bounding box of the reference outline
    min_x, max_x = reference_outline_centered[:, 0].min(
    ), reference_outline_centered[:, 0].max()
    min_y, max_y = reference_outline_centered[:, 1].min(
    ), reference_outline_centered[:, 1].max()

    # Generate grid of points (pixels) covering the bounding box
    x_grid = np.linspace(min_x, max_x, resolution)
    y_grid = np.linspace(min_y, max_y, resolution)
    xv, yv = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([xv.ravel(), yv.ravel()]).T

    # Create Path objects for the reference and target outlines
    reference_path = Path(reference_outline_centered)
    target_path = Path(target_outline_centered)

    # Check which points of the grid are inside both outlines
    points_in_reference = reference_path.contains_points(grid_points)
    points_in_target = target_path.contains_points(grid_points)

    # Calculate the overlap area as the number of points (pixels) inside both outlines
    overlap_area = np.sum(points_in_reference & points_in_target)

    return overlap_area


def rotate_points(points, angle):
    """
    Rotate a set of points by a given angle.

    Parameters:
    points : np.array of shape (n, 2)
    angle : float, angle to rotate by in radians

    Returns:
    rotated_points : np.array of shape (n, 2)
    """
    # Ensure points are a 2D array of shape (N, 2)
    points = np.asarray(points).reshape(-1, 2)

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # Rotate points
    rotated_points = points.dot(rotation_matrix)

    return rotated_points


def grid_search_rotation(reference_outline, target_outline, angle_step=np.pi/36):
    """
    Perform a grid search over possible rotation angles to maximize overlap.

    Parameters:
    reference_outline : np.array of shape (n, 2), outline of the reference bone
    target_outline : np.array of shape (n, 2), outline of the target bone
    angle_step : float, step size for angle search (in radians)

    Returns:
    best_rotated_points : np.array of shape (n, 2), the rotated target bone outline with maximum overlap
    best_angle : float, the optimal rotation angle in radians
    """
    best_angle = None
    max_overlap = -np.inf
    best_rotated_points = None

    # Iterate over angles between -90 and 90 degrees (in radians)
    for angle in np.arange(-np.pi/4, np.pi/4, angle_step):
        rotated_outline = rotate_points(target_outline, angle)
        overlap = calculate_overlap_area(reference_outline, rotated_outline)

        if overlap > max_overlap:
            max_overlap = overlap
            best_angle = angle
            best_rotated_points = rotated_outline

    # Recenter the final rotated outline
    final_centroid = calculate_centroid(best_rotated_points)
    best_rotated_points = translate_to_origin(
        best_rotated_points, final_centroid)

    return best_rotated_points, best_angle, final_centroid

# Modify the process_and_align_bones function to accept the max_width and max_height


def process_and_align_bones_with_overlap(bone_dict, reference_bone, window_size=500, source_col="DAPI"):
    """
    Process and align all bones from the given dictionary to maximize overlap with a reference bone.

    Parameters:
    bone_dict : dict of bones, where each value is a DataFrame with "Position.X" and "Position.Y".
    reference_bone_name : string, the name of the bone to use as the reference for alignment.

    Returns:
    aligned_bones : dict of aligned bone outlines
    """
    aligned_bones = {}
    aligned_angles = {}
    aligned_centroids = {}
    # Check the type of the reference_bone_name
    # If it is a dataframe, we can use the reference_bone_name directly
    if isinstance(reference_bone, pd.DataFrame):
        reference_df = reference_bone
        reference_points = reference_df[reference_df["source"] == source_col][[
            "Position.X", "Position.Y"]].values
        reference_outline = find_outline(
            reference_points, window_size=window_size)
        reference_centroid = calculate_centroid(reference_outline)
        reference_outline = translate_to_origin(
            reference_outline, reference_centroid)

        for bone_name, df in bone_dict.items():

            # Filter points where "source" == "Bone"
            bone_points = df[df["source"] == source_col][[
                "Position.X", "Position.Y"]].values

            # Find the outline of the target bone
            target_outline = find_outline(bone_points, window_size=window_size)
            # Optimize rotation to maximize overlap with the reference bone
            aligned_outline, best_angle, final_centroid = grid_search_rotation(
                reference_outline, target_outline)
            aligned_bones[bone_name] = aligned_outline
            aligned_angles[bone_name] = best_angle
            aligned_centroids[bone_name] = final_centroid
    else:
        raise ValueError("Invalid reference_bone_name. Must be a DataFrame.")

    return aligned_bones, aligned_angles, aligned_centroids

# Perform the alignment for each age group


def align_bones_with_centroids_angles(positions_df, aligned_centroids, aligned_angles, exclude_col="DAPI"):
    aligned_bones = {}

    for bone_name, df in positions_df.items():
        # Filter out rows based on exclude_col first
        # Copy only the filtered rows
        df_filtered = df[df["source"] != exclude_col].copy()

        # Get the centroid and angle for alignment
        centroid = aligned_centroids[bone_name]
        angle = aligned_angles[bone_name]

        # Rotate points
        df_filtered[["Position.X", "Position.Y"]] = rotate_points(
            df_filtered[["Position.X", "Position.Y"]].values, angle)

        # Translate points
        df_filtered["Position.X"] -= centroid[0]
        df_filtered["Position.Y"] -= centroid[1]

        # Save the aligned DataFrame
        aligned_bones[bone_name] = df_filtered

    return aligned_bones


# 2. Functions for bone smoothing
def smooth_outline(outline, sigma=2):
    if not np.array_equal(outline[0], outline[-1]):
        outline = np.vstack([outline, outline[0]])
    smoothed_x = gaussian_filter1d(outline[:, 0], sigma=sigma)
    smoothed_y = gaussian_filter1d(outline[:, 1], sigma=sigma)
    smoothed_outline = np.vstack((smoothed_x, smoothed_y)).T
    if not np.array_equal(smoothed_outline[0], smoothed_outline[-1]):
        smoothed_outline = np.vstack([smoothed_outline, smoothed_outline[0]])
    return smoothed_outline


# 3. Functions for bone transformation
# Get the mask of the bone outline
def points_in_polygon(x_points, y_points, outline):
    path = Path(outline)
    points = np.vstack((x_points, y_points)).T
    return path.contains_points(points)


# Define the function to exclude the data points outside the bone outline
def exclude_outside_bone_outline(df, bone_outline, exclude_source="DAPI"):
    """
    Exclude the data points outside the bone outline using GeoPandas with bounding box filtering for more efficiency.

    Parameters:
    df : DataFrame containing "Position.X", "Position.Y", "weights", and "source".
    bone_outline : The outline of the bone to limit the KDE calculation within the bone.
    exclude_source : The name of the source that used for the bone outline creation.

    Returns:
    df_inside : DataFrame containing the data points inside the bone outline.
    """
    # print(df.shape)
    df = pd.DataFrame(df[df["source"] != exclude_source])
    # print(df.shape)
    # Create a GeoDataFrame from the original DataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(
        df["Position.X"], df["Position.Y"]))

    # Convert the bone outline to a Shapely polygon
    bone_polygon = Polygon(bone_outline)

    # Create a bounding box polygon from the bounds of the bone_polygon
    minx, miny, maxx, maxy = bone_polygon.bounds
    bounding_box = box(minx, miny, maxx, maxy)

    # First, filter by the bounding box of the polygon (faster operation)
    gdf_in_bbox = gdf[gdf.geometry.within(bounding_box)]

    # Then, perform the more precise filtering with the actual polygon
    gdf_inside = gdf_in_bbox[gdf_in_bbox.within(bone_polygon)]

    # Drop the "geometry" column if you don"t need it in the result
    df_inside = gdf_inside.drop(columns="geometry")

    return df_inside


def get_y_range_at_x(shape_points, x):
    """
    Find the range of y-values where the vertical line at x intersects the shape.
    """
    # Find all edges of the shape where x is between the x-coordinates of the endpoints
    y_vals = []
    for i in range(len(shape_points)):
        p1 = shape_points[i]
        # wrap around the shape points
        p2 = shape_points[(i + 1) % len(shape_points)]

        # Check if the x value is between p1 and p2"s x-coordinates
        if (p1[0] <= x <= p2[0]) or (p2[0] <= x <= p1[0]):
            # Linearly interpolate to find the corresponding y value at x
            if p1[0] != p2[0]:  # Avoid division by zero
                y = p1[1] + (p2[1] - p1[1]) * (x - p1[0]) / (p2[0] - p1[0])
                y_vals.append(y)

    if y_vals:
        return min(y_vals), max(y_vals)
    else:
        return None, None  # No intersection with the shape at this x


def create_structured_grid(shape_points, x_num, y_num):
    """
    Create a structured grid by dividing the bounding box of the shape into x_num vertical sections.
    Then place y_num points along each vertical grid line where it intersects the shape.
    """
    shape_points = np.array(shape_points)

    # Step 1: Compute the bounding box
    min_x, max_x = np.min(shape_points[:, 0]), np.max(shape_points[:, 0])

    # Step 2: Divide the x-range into equal sections
    # x_num divisions create x_num + 1 grid lines
    x_vals = np.linspace(min_x, max_x, x_num + 1)
    # Shift the x_vals to the left by half the grid spacing to center the grid
    x_vals = x_vals[1:]  # Remove the first point (left edge of bounding box)
    # Shift left by half the grid spacing
    x_vals = x_vals - (x_vals[1] - x_vals[0]) / 2

    grid_points = []

    # Step 3: For each x grid line, find the y range and then place points
    # Skip the first and last lines (already have the bounding box)
    for x in x_vals:
        y_min, y_max = get_y_range_at_x(shape_points, x)

        if y_min is not None and y_max is not None:
            # Get y points by placing y_num points between y_min and y_max
            y_vals = np.linspace(y_min, y_max, y_num+1)

            # Shift the y_vals down by half the grid spacing to center the grid
            # Remove the first point (bottom edge of bounding box)
            y_vals = y_vals[1:]
            # Shift down by half the grid spacing
            y_vals = y_vals - (y_vals[1] - y_vals[0]) / 2
            # Add grid points (x, y) for this vertical line
            # Skip the first and last points (already have the y_min and y_max)
            for y in y_vals:
                grid_points.append([x, y])

    return np.array(grid_points)


def is_point_inside_shape(point, shape_points):
    """
    Determines if a point is inside an irregular shape using ray-casting.
    """
    x, y = point
    n = len(shape_points)
    inside = False
    p1x, p1y = shape_points[0]
    for i in range(n + 1):
        p2x, p2y = shape_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def thin_plate_spline_transform(src_points, dst_points):
    """
    Perform Thin Plate Spline (TPS) transformation from src_points to dst_points.
    """
    # Create Radial Basis Function (RBF) interpolators for x and y coordinates
    rbf_x = Rbf(src_points[:, 0], src_points[:, 1],
                dst_points[:, 0], function="thin_plate")
    rbf_y = Rbf(src_points[:, 0], src_points[:, 1],
                dst_points[:, 1], function="thin_plate")

    def transform(points):
        new_x = rbf_x(points[:, 0], points[:, 1])
        new_y = rbf_y(points[:, 0], points[:, 1])
        return np.vstack([new_x, new_y]).T

    return transform


def transform_data(data_points, grid_shape_1, grid_shape_2):
    """
    Apply the TPS transformation to the data points based on the grid transformation.
    """
    # Perform Thin Plate Spline (TPS) transformation
    tps_transform = thin_plate_spline_transform(grid_shape_2, grid_shape_1)

    # Apply the transformation to the data points
    transformed_data_points = tps_transform(data_points)

    return transformed_data_points


def filter_bone_positions(df, source_value=None, columns_to_keep=None):
    """
    Filters the DataFrame for a specific source if provided and returns the Position.X, Position.Y columns as a NumPy array.
    If source_value is None, return all positions.
    """
    if source_value is not None:
        filtered_df = df[df["source"] != source_value]
    else:
        # Exclude the data with source value "DAPI"
        filtered_df = df[df["source"] != "DAPI"]
    positions = filtered_df[["Position.X", "Position.Y"]].to_numpy()
    if columns_to_keep is not None:
        return positions, filtered_df["source"].to_numpy(), filtered_df[columns_to_keep].to_numpy()
    else:
        # Return positions and the source column
        return positions, filtered_df["source"].to_numpy()


def transform_bone_positions(outline_dict, position_dict, common_outline, x_num=40, y_num=20, source_value=None, columns_to_keep=None):
    """
    Transforms the bone positions from multiple datasets using Thin Plate Spline (TPS) based on the provided outlines and positions.
    Parameters:
        outline_dict: Dictionary containing outlines.
        position_dict: Dictionary containing bone positions (DataFrames).
        common_outline: The common outline (to which the other outlines will be aligned).
        x_num: Number of vertical sections for structured grid.
        y_num: Number of horizontal points along each vertical section.
        source_value: If provided, exclude the data with this source value.
        columns_to_keep: If provided, keep the specified columns in the transformed DataFrame.
    Returns:
        Dictionary containing the transformed bone positions with the "source" column retained.
    """
    transformed_dict = {}

    # Create the structured grid for the common outline
    grid_common_outline = create_structured_grid(
        common_outline, x_num=x_num, y_num=y_num)

    # Loop through each dataset in the position_dict
    for dataset_name, position_df in position_dict.items():
        # Get the corresponding outline
        outline_2 = outline_dict[dataset_name]

        # Create the structured grid for the specific dataset"s outline
        grid_outline_2 = create_structured_grid(
            outline_2, x_num=x_num, y_num=y_num)

        # Filter the positions based on the source (if provided)
        # By default, it will exclude the data with source value "GFP"
        if columns_to_keep is None:
            bone_positions_2, source_column = filter_bone_positions(
                position_df, source_value=source_value)
        else:
            bone_positions_2, source_column, kept_columns = filter_bone_positions(
                position_df, source_value=source_value, columns_to_keep=columns_to_keep)

        # Transform the filtered bone positions from the dataset outline to the common outline
        transformed_bone_positions = transform_data(
            bone_positions_2, grid_common_outline, grid_outline_2)

        # Convert the transformed positions to a DataFrame and include the source column
        transformed_df = pd.DataFrame(transformed_bone_positions, columns=[
            "Position.X", "Position.Y"])
        transformed_df["source"] = source_column  # Add the source column back
        # Add the dataset name for reference
        transformed_df["dataset"] = dataset_name
        if columns_to_keep is not None:
            transformed_df[columns_to_keep] = kept_columns
        # Store the transformed DataFrame in the result dictionary
        transformed_dict[dataset_name] = transformed_df

    return transformed_dict
