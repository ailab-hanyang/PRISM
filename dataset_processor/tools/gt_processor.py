"""
GT Preprocessing for Lab Dataset (Camera Coordinate System)
Transform from LiDAR coordinate to Camera ground-projected coordinate
"""

import numpy as np
import open3d as o3d
import yaml
import torch
from pathlib import Path


def create_rotation_matrix(roll_deg, pitch_deg):
    """
    Create rotation matrix for roll (x-axis) and pitch (y-axis) correction

    Args:
        roll_deg: Roll angle in degrees (rotation around x-axis)
        pitch_deg: Pitch angle in degrees (rotation around y-axis)

    Returns:
        3x3 rotation matrix
    """
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)

    # Rotation matrix around x-axis (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Rotation matrix around y-axis (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Combined rotation: first roll, then pitch
    return Ry @ Rx


class GTProcessor:
    """
    GT generation in camera coordinate system (ground-projected view).
    Performs LiDAR -> Camera coordinate transformation.
    """

    # LiDAR ground plane correction (roll: 0.58°, pitch: 0.67°)
    LIDAR_ROTATION_MATRIX = create_rotation_matrix(roll_deg=0.58, pitch_deg=0.67)

    def __init__(self, config_path):
        """
        Args:
            config_path: path to config.yaml
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Coordinate transform parameters (in meters)
        transform = self.config.get('coordinate_transform', {})
        self.offset_x = transform.get('offset_x', 0.0)  # m
        self.offset_y = transform.get('offset_y', 0.0)  # m
        self.offset_z = transform.get('offset_z', 0.0)  # m

        # ROI settings (in cm)
        # Camera coordinate system: x=forward(longitudinal), y=lateral, z=height(vertical)
        roi = self.config['roi']
        self.roi_x_min_config = roi['x_min']  # initial x_min from config
        self.roi_x_max_config = roi['x_max']  # initial x_max from config
        self.roi_length = self.roi_x_max_config - self.roi_x_min_config  # ROI length (fixed)

        # Actual ROI in use (adjusted by dx)
        self.roi_x_min = self.roi_x_min_config
        self.roi_x_max = self.roi_x_max_config
        self.roi_y_min = roi['y_min']  # lateral min
        self.roi_y_max = roi['y_max']  # lateral max
        self.roi_z_min = roi['z_min']  # height min
        self.roi_z_max = roi['z_max']  # height max

        # Grid resolution (cm)
        grid = self.config['grid']
        self.grid_res_x = grid['resolution_x']  # forward
        self.grid_res_y = grid['resolution_y']  # lateral
        self.grid_res_z = grid['resolution_z']  # height

        # Grid count (X=forward, Y=lateral)
        self.num_grids_x = int((self.roi_x_max - self.roi_x_min) / self.grid_res_x)
        self.num_grids_y = int((self.roi_y_max - self.roi_y_min) / self.grid_res_y)
        self.num_grids_z = int((self.roi_z_max - self.roi_z_min) / self.grid_res_z)

        self.include_normal_gt = self.config.get('include_normal_gt', False)
        self.gt_folder_name = self.config.get('gt_folder_name', 'elevation_map')

    def update_roi_x(self, dx):
        """
        Update ROI X (forward) range by shifting the entire range by dx.

        Args:
            dx: X-axis offset (cm, 0~600)
        """
        # Shift entire ROI by dx (length remains fixed)
        self.roi_x_min = self.roi_x_min_config + dx
        self.roi_x_max = self.roi_x_max_config + dx

        # Grid count stays fixed (ROI length is constant)
        self.num_grids_x = int(self.roi_length / self.grid_res_x)

    def load_pcd(self, pcd_path):
        """
        Load point cloud and transform coordinates (LiDAR -> Camera).

        Args:
            pcd_path: path to .pcd file

        Returns:
            cloud: open3d.geometry.PointCloud (camera coordinate system)
        """
        cloud = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(cloud.points)

        # Coordinate transform: LiDAR -> Camera
        points_transformed = points.copy()
        points_transformed[:, 0] += self.offset_x  # x: keep
        points_transformed[:, 1] += self.offset_y  # y: keep
        points_transformed[:, 2] += self.offset_z  # z: +1.8m

        cloud_transformed = o3d.geometry.PointCloud()
        cloud_transformed.points = o3d.utility.Vector3dVector(points_transformed)

        return cloud_transformed

    def crop_roi(self, cloud, apply_ground_correction=True):
        """
        Crop ROI and apply ground correction (in camera coordinate system).
        Crops ROI first, then applies roll/pitch correction to reduce computation.

        Args:
            cloud: open3d.geometry.PointCloud (camera coordinate system)
            apply_ground_correction: whether to apply roll/pitch ground correction

        Returns:
            cloud_roi: cropped and corrected point cloud
        """
        points = np.asarray(cloud.points)

        # cm -> m conversion
        # Camera coordinate: x=forward, y=lateral, z=height(ground-relative)
        x_min = self.roi_x_min / 100.0  # forward start
        x_max = self.roi_x_max / 100.0  # forward end
        y_min = self.roi_y_min / 100.0  # lateral min
        y_max = self.roi_y_max / 100.0  # lateral max
        z_min = self.roi_z_min / 100.0  # height min
        z_max = self.roi_z_max / 100.0  # height max

        # ROI mask (Z filter removed — use all height points)
        xy_mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &  # X: forward
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max)  # Y: lateral
        )

        # Crop
        cropped_points = points[xy_mask]

        # LiDAR ground correction (roll: 0.58°, pitch: 0.67°)
        # applied after ROI crop to reduce computation
        if apply_ground_correction and len(cropped_points) > 0:
            cropped_points = (self.LIDAR_ROTATION_MATRIX @ cropped_points.T).T

        z_mask = (
            (cropped_points[:, 2] >= z_min) & (cropped_points[:, 2] <= z_max)    # Z: height
        )
        cropped_points = cropped_points[z_mask]

        cloud_roi = o3d.geometry.PointCloud()
        cloud_roi.points = o3d.utility.Vector3dVector(cropped_points)

        return cloud_roi

    def filter_dynamic_objects(self, cloud_roi, grid_res=0.2, n_std=3.0,
                               remove_noise=True, nb_neighbors=20, std_ratio=2.0):
        """
        Remove dynamic objects (vehicles, pedestrians, etc.) while preserving ground objects (speed bumps, potholes).

        Removes outliers outside median +/- n_std * std per grid cell.
        - If no dynamic objects, almost all points are kept.
        - Dynamic objects are removed as statistical outliers.
        - Speed bumps/potholes are preserved as ground-level height variations.

        Followed by Statistical Outlier Removal to eliminate scattered airborne noise points.

        Args:
            cloud_roi: ROI-cropped point cloud
            grid_res: grid resolution (m), default 0.2m (20cm)
            n_std: outlier threshold (median +/- n_std * std), default 3.0
            remove_noise: whether to apply Statistical Outlier Removal for scattered noise
            nb_neighbors: SOR neighbor count, default 20
            std_ratio: SOR distance threshold (remove if > mean + std_ratio * std), default 2.0

        Returns:
            cloud_filtered: point cloud with dynamic objects and noise removed
        """
        points = np.asarray(cloud_roi.points)

        if len(points) == 0:
            return cloud_roi

        # Grid index computation
        grid_x = np.floor(points[:, 0] / grid_res).astype(int)
        grid_y = np.floor(points[:, 1] / grid_res).astype(int)

        # Remove outliers per grid cell
        filtered_indices = []

        # Unique grid cells
        unique_grids = np.unique(np.column_stack([grid_x, grid_y]), axis=0)

        for gx, gy in unique_grids:
            # Points in this grid cell
            cell_mask = (grid_x == gx) & (grid_y == gy)
            cell_indices = np.where(cell_mask)[0]
            cell_z = points[cell_indices, 2]

            if len(cell_z) > 0:
                # Compute median and std
                z_median = np.median(cell_z)
                z_std = np.std(cell_z)

                # Keep only points within median +/- n_std * std
                # Only upper bound applied since we target above-ground objects (vehicles, people)
                z_upper = z_median + n_std * z_std
                inlier_mask = cell_z <= z_upper

                filtered_indices.extend(cell_indices[inlier_mask])

        if len(filtered_indices) == 0:
            return cloud_roi

        filtered_points = points[filtered_indices]

        cloud_filtered = o3d.geometry.PointCloud()
        cloud_filtered.points = o3d.utility.Vector3dVector(filtered_points)

        # Statistical Outlier Removal: remove sparse noise points with few neighbors
        if remove_noise and len(filtered_points) > nb_neighbors:
            cloud_filtered, _ = cloud_filtered.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio
            )

        return cloud_filtered

    def filter_road_by_elevation_histogram(self, ele_gt, ele_mask,
                                           bin_size=2.0,
                                           min_peak_ratio=0.15):
        """
        Extract road surface only via elevation histogram analysis (removes sidewalks, curbs, trees).

        Approach:
        - Build histogram of elevation values and detect peaks.
        - Single peak or all similar heights -> keep all (flat road).
        - Multiple peaks (bimodal/multimodal) -> find lowest peak (road) and nearest other peak.
        - Set threshold at midpoint between the two peaks; keep only values below.

        Advantages:
        - Flat roads are untouched (no false positives).
        - Filters automatically only when curbs/sidewalks are present.
        - Threshold is determined data-adaptively.

        Args:
            ele_gt: [num_grids_x, num_grids_y] elevation map (cm)
            ele_mask: [num_grids_x, num_grids_y] validity mask
            bin_size: histogram bin size (cm), default 2.0cm
            min_peak_ratio: minimum ratio to max peak to be considered a peak, default 0.15

        Returns:
            filtered_mask: [num_grids_x, num_grids_y] new mask containing road only
        """
        # Extract valid elevation values only
        valid_elevations = ele_gt[ele_mask].cpu().numpy()

        if len(valid_elevations) < 10:
            # Too few data points — return original
            return ele_mask

        # Build histogram
        ele_min = valid_elevations.min()
        ele_max = valid_elevations.max()
        num_bins = int(np.ceil((ele_max - ele_min) / bin_size))

        if num_bins < 3:
            # Negligible height variation (flat road) — keep all
            return ele_mask

        hist, bin_edges = np.histogram(valid_elevations, bins=num_bins, range=(ele_min, ele_max))

        # Peak detection (find local maxima)
        peaks = []
        peak_heights = []

        for i in range(1, len(hist) - 1):
            # Local maximum: higher than both neighbors
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peak_height = hist[i]
                # Minimum height condition (at least min_peak_ratio of global max peak)
                if peak_height >= hist.max() * min_peak_ratio:
                    peak_elevation = (bin_edges[i] + bin_edges[i+1]) / 2.0
                    peaks.append(peak_elevation)
                    peak_heights.append(peak_height)

        if len(peaks) <= 1:
            # Single peak or fewer — unimodal (flat road) — keep all
            return ele_mask

        # Multiple peaks — find lowest peak (road) and its nearest neighbor
        peaks = np.array(peaks)
        peak_heights = np.array(peak_heights)

        # Lowest peak = road
        lowest_peak_idx = np.argmin(peaks)
        road_peak_elevation = peaks[lowest_peak_idx]

        # Remaining peaks excluding the road peak
        other_peaks = np.delete(peaks, lowest_peak_idx)

        # Find peak nearest to road peak (sidewalk/curb)
        distances = np.abs(other_peaks - road_peak_elevation)
        nearest_peak_idx = np.argmin(distances)
        nearest_peak_elevation = other_peaks[nearest_peak_idx]

        # Set threshold at midpoint between the two peaks
        threshold = (road_peak_elevation + nearest_peak_elevation) / 2.0

        # Keep only values below threshold
        filtered_mask = ele_mask.clone()
        elevation_below_threshold = ele_gt <= threshold
        filtered_mask = filtered_mask & elevation_below_threshold

        return filtered_mask

    def remove_isolated_outliers(self, ele_gt, ele_mask, z_std_threshold=3.0):
        """
        Remove isolated outlier grids remaining after histogram filtering.

        Approach:
        - Compare each grid to its 8 neighbors.
        - Remove if outside median +/- z_std_threshold * std of neighbors.
        - Eliminates isolated high values (sensor noise, birds, etc.).

        Args:
            ele_gt: [num_grids_x, num_grids_y] elevation map (cm)
            ele_mask: [num_grids_x, num_grids_y] validity mask
            z_std_threshold: std multiplier (default 3.0)

        Returns:
            filtered_mask: [num_grids_x, num_grids_y] mask with outliers removed
        """
        filtered_mask = ele_mask.clone()

        # Convert to NumPy for efficient computation
        ele_np = ele_gt.cpu().numpy()
        mask_np = ele_mask.cpu().numpy()

        # Iterate grids and compare to neighbors
        for x in range(self.num_grids_x):
            for y in range(self.num_grids_y):
                if not mask_np[x, y]:
                    continue

                current_elevation = ele_np[x, y]

                # Collect 8-neighbor grids
                neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue  # skip self

                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.num_grids_x and 0 <= ny < self.num_grids_y:
                            if mask_np[nx, ny]:
                                neighbors.append(ele_np[nx, ny])

                # Only check when enough neighbors exist
                if len(neighbors) >= 3:
                    neighbor_median = np.median(neighbors)
                    neighbor_std = np.std(neighbors)

                    # Compute neighbor range
                    lower_bound = neighbor_median - z_std_threshold * neighbor_std
                    upper_bound = neighbor_median + z_std_threshold * neighbor_std

                    # Remove if outside range
                    if current_elevation < lower_bound or current_elevation > upper_bound:
                        filtered_mask[x, y] = False

        return filtered_mask

    def get_gt_elevation(self, cloud_roi):
        """
        Elevation Map generation (same logic as RoadBEV).

        Args:
            cloud_roi: ROI-cropped point cloud

        Returns:
            ele_gt: [num_grids_x, num_grids_y] elevation map (cm) - [X(longitudinal), Y(lateral)] RSRD format
            ele_mask: [num_grids_x, num_grids_y] validity mask
        """
        points = np.asarray(cloud_roi.points)

        if len(points) == 0:
            return (
                torch.zeros((self.num_grids_x, self.num_grids_y), dtype=torch.float32),
                torch.zeros((self.num_grids_x, self.num_grids_y), dtype=torch.bool)
            )

        # m -> cm conversion
        points_cm = points * 100.0

        # LiDAR coordinate: x=forward, y=lateral, z=height
        points_xy = points_cm[:, [0, 1]]  # [N, 2] (forward, lateral)
        points_z = points_cm[:, 2]        # [N] (height)

        # Grid initialization [X(forward), Y(lateral)] - same RSRD format [164, 64]
        grids_z = torch.zeros((self.num_grids_x, self.num_grids_y), dtype=torch.float32)
        grids_count = torch.zeros((self.num_grids_x, self.num_grids_y), dtype=torch.int32)

        # Assign each point to a grid
        for xy, z in zip(points_xy, points_z):
            idx_x = int((xy[0] - self.roi_x_min) / self.grid_res_x)  # forward
            idx_y = int((xy[1] - self.roi_y_min) / self.grid_res_y)  # lateral

            if 0 <= idx_x < self.num_grids_x and 0 <= idx_y < self.num_grids_y:
                grids_z[idx_x, idx_y] += z  # stored as [X, Y] (RSRD format)
                grids_count[idx_x, idx_y] += 1

        # Build mask (True only for grids with points)
        mask = grids_count > 0

        # Compute average height
        grids_z[mask] = grids_z[mask] / grids_count[mask].float()

        return grids_z, mask

    def get_gt_normal_vector(self, cloud_roi, min_points=3):
        """
        Compute GT normal vectors per grid cell via PCA (X, Y, Z directions).

        Applies PCA to points in each cell; the eigenvector with the smallest eigenvalue
        (minimum variance direction) is used as the normal vector.
        Direction is unified so that the Z component is always positive.

        Args:
            cloud_roi: ROI-cropped point cloud
            min_points: minimum number of points required for normal estimation (default 3)

        Returns:
            normal_gt: [num_grids_x, num_grids_y, 3] normal vector map - [X(longitudinal), Y(lateral), Z(vertical)]
            normal_mask: [num_grids_x, num_grids_y] validity mask
        """
        points = np.asarray(cloud_roi.points)

        normal_gt = torch.zeros((self.num_grids_x, self.num_grids_y, 3), dtype=torch.float32)
        normal_mask = torch.zeros((self.num_grids_x, self.num_grids_y), dtype=torch.bool)

        if len(points) == 0:
            return normal_gt, normal_mask

        # m -> cm conversion
        points_cm = points * 100.0

        # Grid index computation
        idx_x = ((points_cm[:, 0] - self.roi_x_min) / self.grid_res_x).astype(int)
        idx_y = ((points_cm[:, 1] - self.roi_y_min) / self.grid_res_y).astype(int)

        # Valid range mask
        valid = (
            (idx_x >= 0) & (idx_x < self.num_grids_x) &
            (idx_y >= 0) & (idx_y < self.num_grids_y)
        )
        points_cm = points_cm[valid]
        idx_x = idx_x[valid]
        idx_y = idx_y[valid]

        if len(points_cm) == 0:
            return normal_gt, normal_mask

        # Group points by cell
        cell_keys = idx_x * self.num_grids_y + idx_y
        sort_order = np.argsort(cell_keys)
        cell_keys_sorted = cell_keys[sort_order]
        points_sorted = points_cm[sort_order]

        unique_keys, group_starts = np.unique(cell_keys_sorted, return_index=True)
        group_ends = np.append(group_starts[1:], len(cell_keys_sorted))

        for key, start, end in zip(unique_keys, group_starts, group_ends):
            cell_points = points_sorted[start:end]  # [M, 3]

            if len(cell_points) < min_points:
                continue

            # PCA: eigenvector of minimum eigenvalue of covariance matrix = normal vector
            centered = cell_points - cell_points.mean(axis=0)
            cov = (centered.T @ centered) / len(cell_points)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order

            # Minimum eigenvalue -> normal vector (eigenvectors in column direction)
            normal = eigenvectors[:, 0]

            # Unify direction so Z component is positive (pointing upward)
            if normal[2] < 0:
                normal = -normal

            gx = key // self.num_grids_y
            gy = key % self.num_grids_y
            normal_gt[gx, gy] = torch.from_numpy(normal.astype(np.float32))
            normal_mask[gx, gy] = True

        return normal_gt, normal_mask


    def process(self, pcd_path, dx=0.0, filter_dynamic=True, n_std=3.0,
                remove_noise=True, nb_neighbors=20, std_ratio=2.0,
                filter_by_histogram=False, hist_bin_size=2.0, min_peak_ratio=0.15,
                remove_outliers=False, z_std_threshold=3.0):
        """
        Full processing pipeline.

        Args:
            pcd_path: path to .pcd file
            dx: ROI X (forward) offset (cm, 0~600)
            filter_dynamic: whether to apply dynamic object filtering (default True)
            n_std: outlier threshold (median +/- n_std * std), default 3.0
                   - 3.0: preserves speed bumps/potholes, removes vehicles/pedestrians
            remove_noise: whether to apply scattered noise removal (default True)
            nb_neighbors: SOR neighbor count (default 20)
            std_ratio: SOR distance threshold (default 2.0)
            filter_by_histogram: whether to apply elevation histogram filtering
                                  (removes sidewalks, curbs, trees) (default False)
            hist_bin_size: histogram bin size (cm) (default 2.0)
            min_peak_ratio: minimum ratio to be considered a peak (default 0.15)
            remove_outliers: whether to apply isolated outlier grid removal (default False)
            z_std_threshold: outlier detection std multiplier (default 3.0)

        Returns:
            ele_gt: [num_grids_y, num_grids_x] elevation map - [Y(lateral), X(longitudinal)]
            ele_mask: [num_grids_y, num_grids_x] validity mask
            cloud_roi: ROI-cropped point cloud
        """
        # Update ROI
        self.update_roi_x(dx)

        # 1. Load PCD
        cloud = self.load_pcd(pcd_path)

        # 2. ROI crop + LiDAR ground correction (roll/pitch)
        cloud_roi = self.crop_roi(cloud)

        # 3. Dynamic object filtering (remove vehicles/pedestrians, preserve bumps/potholes) + noise removal
        if filter_dynamic:
            cloud_roi = self.filter_dynamic_objects(
                cloud_roi, n_std=n_std,
                remove_noise=remove_noise,
                nb_neighbors=nb_neighbors, std_ratio=std_ratio
            )

        # 4. Generate elevation map
        ele_gt, ele_mask = self.get_gt_elevation(cloud_roi)

        # 5. Elevation histogram-based filtering (remove curbs/sidewalks/trees) — optional
        if filter_by_histogram:
            ele_mask = self.filter_road_by_elevation_histogram(
                ele_gt, ele_mask,
                bin_size=hist_bin_size,
                min_peak_ratio=min_peak_ratio
            )

        # 6. Remove isolated outlier grids (post-processing) — optional
        if remove_outliers:
            ele_mask = self.remove_isolated_outliers(
                ele_gt, ele_mask,
                z_std_threshold=z_std_threshold
            )

        if not self.include_normal_gt:
            return ele_gt, ele_mask, cloud_roi

        normal_gt, normal_mask = self.get_gt_normal_vector(cloud_roi)
        return ele_gt, ele_mask, normal_gt, normal_mask, cloud_roi


def main():
    """Test / demo entry point."""
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description='Lab GT Preprocessing (LiDAR Local)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='config.yaml path')
    parser.add_argument('--pcd', type=str, default=None,
                        help='input .pcd file (overrides config)')
    parser.add_argument('--dx', type=float, default=0.0,
                        help='ROI Z offset in cm (0~600)')
    parser.add_argument('--save', type=str, default=None,
                        help='output .pkl file path')
    parser.add_argument('--no-filter', action='store_true',
                        help='Disable dynamic object filtering')
    parser.add_argument('--n-std', type=float, default=3.0,
                        help='Outlier threshold (median + n_std * std), default: 3.0')
    parser.add_argument('--no-noise-removal', action='store_true',
                        help='Disable statistical outlier removal for scattered noise')
    parser.add_argument('--nb-neighbors', type=int, default=20,
                        help='SOR: number of neighbors to consider, default: 20')
    parser.add_argument('--std-ratio', type=float, default=2.0,
                        help='SOR: std ratio threshold, default: 2.0')
    parser.add_argument('--filter-by-histogram', action='store_true',
                        help='Enable elevation histogram-based filtering (remove sidewalks, curbs, trees)')
    parser.add_argument('--hist-bin-size', type=float, default=2.0,
                        help='Histogram bin size in cm, default: 2.0')
    parser.add_argument('--min-peak-ratio', type=float, default=0.15,
                        help='Minimum peak ratio to max peak, default: 0.15 (15%%)')
    parser.add_argument('--remove-outliers', action='store_true',
                        help='Enable isolated outlier grid removal (post-processing)')
    parser.add_argument('--z-std-threshold', type=float, default=3.0,
                        help='Outlier detection std threshold, default: 3.0')
    args = parser.parse_args()

    # Initialize processor
    processor = GTProcessor(args.config)

    # PCD path
    if args.pcd is None:
        pcd_path = processor.config['input']['lidar_pcd']
    else:
        pcd_path = args.pcd

    # Process
    print(f"\n{'='*60}")
    print(f"Processing: {pcd_path}")
    print(f"dx offset: {args.dx} cm")
    print(f"Ground correction: roll=0.58°, pitch=0.67°")
    print(f"Dynamic object filtering: {not args.no_filter}")
    if not args.no_filter:
        print(f"  Outlier threshold: median + {args.n_std} * std")
        print(f"  Noise removal (SOR): {not args.no_noise_removal}")
        if not args.no_noise_removal:
            print(f"    nb_neighbors={args.nb_neighbors}, std_ratio={args.std_ratio}")
    print(f"Histogram-based filtering: {args.filter_by_histogram}")
    if args.filter_by_histogram:
        print(f"  hist_bin_size={args.hist_bin_size}cm, min_peak_ratio={args.min_peak_ratio}")
    print(f"Isolated outlier removal: {args.remove_outliers}")
    if args.remove_outliers:
        print(f"  z_std_threshold={args.z_std_threshold}")
    print(f"{'='*60}\n")

    ele_gt, ele_mask, cloud_roi = processor.process(
        pcd_path, dx=args.dx,
        filter_dynamic=not args.no_filter,
        n_std=args.n_std,
        remove_noise=not args.no_noise_removal,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        filter_by_histogram=args.filter_by_histogram,
        hist_bin_size=args.hist_bin_size,
        min_peak_ratio=args.min_peak_ratio,
        remove_outliers=args.remove_outliers,
        z_std_threshold=args.z_std_threshold
    )

    # Save
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump([ele_gt, ele_mask], f)

        print(f"\n[GTProcessor] Saved to: {save_path}")

    print(f"\n{'='*60}")
    print(f"Final Grid: {ele_gt.shape}")
    print(f"Valid cells: {ele_mask.sum()}/{ele_mask.numel()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
