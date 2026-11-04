"""
Visualize PKL in 6-panel format.
"""
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import yaml
from pathlib import Path
import sys


def create_rotation_matrix(roll_deg, pitch_deg):
    """Create rotation matrix for roll (x-axis) and pitch (y-axis) correction"""
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    return Ry @ Rx


# LiDAR ground plane correction (roll: 0.58°, pitch: 0.67°)
LIDAR_ROTATION_MATRIX = create_rotation_matrix(roll_deg=0.58, pitch_deg=0.67)


def visualize_pkl_6panel(pkl_path, pcd_path, output_path):
    """Visualize PKL in 6-panel format."""

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # ROI settings
    roi = config['roi']
    transform = config.get('coordinate_transform', {})
    offset_x = transform.get('offset_x', 0.0)
    offset_z = transform.get('offset_z', 0.0)

    roi_x_min = roi['x_min'] / 100.0
    roi_x_max = roi['x_max'] / 100.0
    roi_y_min = roi['y_min'] / 100.0
    roi_y_max = roi['y_max'] / 100.0
    roi_z_min = roi['z_min'] / 100.0
    roi_z_max = roi['z_max'] / 100.0

    # Load PKL
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    ele_gt = data[0]
    ele_mask = data[1]

    # Load and transform PCD
    cloud = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(cloud.points)
    points_transformed = points.copy()
    points_transformed[:, 0] += offset_x
    points_transformed[:, 2] += offset_z

    # ROI crop
    xy_mask = (
        (points_transformed[:, 0] >= roi_x_min) & (points_transformed[:, 0] <= roi_x_max) &
        (points_transformed[:, 1] >= roi_y_min) & (points_transformed[:, 1] <= roi_y_max)
    )
    points_roi = points_transformed[xy_mask]
    
    # Ground plane correction (roll: 0.58°, pitch: 0.67°) - same as preprocess_gt_lab.py
    if len(points_roi) > 0:
        points_roi = (LIDAR_ROTATION_MATRIX @ points_roi.T).T
        
    z_mask = (
        (points_roi[:, 2] >= roi_z_min) & (points_roi[:, 2] <= roi_z_max)    # Z: height
    )
    points_roi = points_roi[z_mask]

    # Create figure
    fig = plt.figure(figsize=(20, 10))
    
    # Title
    title_text = f'PKL: {Path(pkl_path).name} | Shape: {ele_gt.shape} | Valid: {ele_mask.sum().item()}/{ele_mask.numel()} ({ele_mask.float().mean()*100:.1f}%)'
    fig.suptitle(title_text, fontsize=14, fontweight='bold')
    
    # 1. Full Point Cloud (Top view)
    ax1 = fig.add_subplot(2, 3, 1)
    sample_idx = np.random.choice(len(points_transformed), min(50000, len(points_transformed)), replace=False)
    points_sample = points_transformed[sample_idx]
    scatter = ax1.scatter(points_sample[:, 0], points_sample[:, 1], 
                         c=points_sample[:, 2], cmap='viridis', s=0.5, alpha=0.6)
    
    # ROI box
    from matplotlib.patches import Rectangle
    rect = Rectangle((roi_x_min, roi_y_min), roi_x_max-roi_x_min, roi_y_max-roi_y_min,
                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax1.add_patch(rect)
    ax1.set_xlabel('X (forward) [m]')
    ax1.set_ylabel('Y (lateral) [m]')
    ax1.set_title(f'Full Point Cloud with ROI\nPoints: {len(points_transformed):,}')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Z (height) [m]', fraction=0.046)
    
    # 2. ROI Point Cloud (Zoomed)
    ax2 = fig.add_subplot(2, 3, 4)
    if len(points_roi) > 0:
        scatter2 = ax2.scatter(points_roi[:, 0], points_roi[:, 1], 
                              c=points_roi[:, 2], cmap='viridis', s=1, alpha=0.8)
        ax2.set_xlim(roi_x_min, roi_x_max)
        ax2.set_ylim(roi_y_min, roi_y_max)
    ax2.set_xlabel('X (forward) [m]')
    ax2.set_ylabel('Y (lateral) [m]')
    ax2.set_title(f'ROI Point Cloud (Zoomed)\nPoints: {len(points_roi):,}')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    if len(points_roi) > 0:
        plt.colorbar(scatter2, ax=ax2, label='Z (height) [m]', fraction=0.046)
    
    # 3. Elevation Map
    ax3 = fig.add_subplot(2, 3, 2)
    ele_gt_vis = ele_gt.numpy().copy()
    ele_gt_vis[~ele_mask.numpy()] = np.nan
    im3 = ax3.imshow(ele_gt_vis.T, cmap='plasma', aspect='auto', origin='lower')
    ax3.set_title(f'Elevation Map\nShape: {ele_gt.shape}')
    ax3.set_xlabel(f'Dim 0 (size={ele_gt.shape[0]})')
    ax3.set_ylabel(f'Dim 1 (size={ele_gt.shape[1]})')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(im3, ax=ax3, label='Elevation (cm)', fraction=0.046)
    
    # 4. Validity Mask
    ax4 = fig.add_subplot(2, 3, 5)
    im4 = ax4.imshow(ele_mask.numpy().T.astype(float), cmap='RdYlGn', aspect='auto', origin='lower', vmin=0, vmax=1)
    ax4.set_title(f'Validity Mask\nValid: {ele_mask.sum().item()}/{ele_mask.numel()} ({ele_mask.float().mean()*100:.1f}%)')
    ax4.set_xlabel(f'Dim 0 (size={ele_gt.shape[0]})')
    ax4.set_ylabel(f'Dim 1 (size={ele_gt.shape[1]})')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(im4, ax=ax4, label='Valid (1) / Invalid (0)', ticks=[0, 1], fraction=0.046)
    
    # 5. Elevation Distribution
    ax5 = fig.add_subplot(2, 3, 3)
    if ele_mask.sum() > 0:
        valid_elevations = ele_gt[ele_mask].numpy()
        ax5.hist(valid_elevations, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax5.axvline(valid_elevations.mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {valid_elevations.mean():.2f} cm')
        ax5.axvline(np.median(valid_elevations), color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(valid_elevations):.2f} cm')
        ax5.set_title(f'Elevation Distribution\nRange: [{valid_elevations.min():.1f}, {valid_elevations.max():.1f}] cm')
        ax5.set_xlabel('Elevation (cm)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Z Distribution in ROI
    ax6 = fig.add_subplot(2, 3, 6)
    if len(points_roi) > 0:
        z_roi = points_roi[:, 2] * 100
        ax6.hist(z_roi, bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
        ax6.axvline(z_roi.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {z_roi.mean():.2f} cm')
        ax6.axvline(np.median(z_roi), color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(z_roi):.2f} cm')
        ax6.axvline(roi_z_min*100, color='black', linestyle=':', linewidth=2, alpha=0.5,
                    label=f'Z min: {roi_z_min*100:.0f} cm')
        ax6.axvline(roi_z_max*100, color='black', linestyle=':', linewidth=2, alpha=0.5,
                    label=f'Z max: {roi_z_max*100:.0f} cm')
        ax6.set_title(f'Z Distribution in ROI\nPoints: {len(points_roi):,}')
        ax6.set_xlabel('Z (height) [cm]')
        ax6.set_ylabel('Frequency')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\n✓ Saved: {output_path}')
    plt.close()

if __name__ == '__main__':
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths from config or command line
    if len(sys.argv) >= 4:
        # Command line arguments (backward compatibility)
        pkl_path = sys.argv[1]
        pcd_path = sys.argv[2]
        output_path = sys.argv[3]
    else:
        # Read from config.yaml
        vis_config = config.get('visualization', {})
        pkl_path = vis_config.get('pkl_path')
        pcd_path = vis_config.get('pcd_path')
        output_path = vis_config.get('output_path', 'visualization_output.png')
        
        if not pkl_path or not pcd_path:
            print("[ERROR] Please specify pkl_path and pcd_path in config.yaml or provide as command line arguments")
            print("Usage: python visualize_pkl_6panel_simple.py [<pkl> <pcd> <output>]")
            sys.exit(1)
        
        print(f"Reading from config.yaml:")
        print(f"  PKL: {pkl_path}")
        print(f"  PCD: {pcd_path}")
        print(f"  Output: {output_path}")
    
    visualize_pkl_6panel(pkl_path, pcd_path, output_path)
