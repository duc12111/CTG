#!/usr/bin/env python3
"""
Script to visualize trajdata rasterized images used during training.
This helps understand what the model sees as input.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import DataLoader

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tbsim.utils.batch_utils import batch_utils, set_global_batch_type
from tbsim.configs.base import ExperimentConfig
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg, verify_map
import tbsim.utils.tensor_utils as TensorUtils
from trajdata.maps.raster_map import RasterizedMap

# Set matplotlib to non-interactive backend to avoid display issues
plt.switch_backend('Agg')


def setup_trajdata_env(config):
    """Setup trajdata environment configuration"""
    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env("nuplan_mini")
    
    # Set up raster configuration
    raster_cfg = {
        "include_hist": config.env.rasterizer.include_hist,
        "pixel_size": config.env.rasterizer.pixel_size,
        "raster_size": config.env.rasterizer.raster_size,
        "ego_center": config.env.rasterizer.ego_center,
        "num_sem_layers": config.env.rasterizer.num_sem_layers,
        "no_map_fill_value": config.env.rasterizer.no_map_fill_value,
        "drivable_layers": config.env.rasterizer.drivable_layers
    }
    set_global_trajdata_batch_raster_cfg(raster_cfg)


def create_dataloader(config):
    """Create a trajdata dataloader for nuPlan mini"""
    # Import the datamodule
    from tbsim.datasets.factory import datamodule_factory
    
    # Create datamodule
    datamodule = datamodule_factory(
        cls_name=config.train.datamodule_class, 
        config=config
    )
    datamodule.setup("fit")
    
    # Get train dataloader
    train_loader = datamodule.train_dataloader()
    
    return train_loader, config


def visualize_raster_channels(image, title="Rasterized Input"):
    """Visualize different channels of the rasterized image"""
    # Convert to numpy if it's a tensor
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    # Get number of channels
    num_channels = image.shape[0]
    
    # Create subplots
    fig, axes = plt.subplots(2, (num_channels + 1) // 2, figsize=(15, 8))
    fig.suptitle(title, fontsize=16)
    
    if num_channels == 1:
        axes = [axes]
    elif num_channels <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each channel
    for i in range(num_channels):
        ax = axes[i] if num_channels > 1 else axes[0]
        
        # Normalize the channel for visualization
        channel_data = image[i]
        
        # Handle different value ranges
        if np.max(channel_data) > 1.0:
            # Likely contains raw values, normalize
            channel_data = (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data) + 1e-8)
        
        im = ax.imshow(channel_data, cmap='viridis', origin='lower')
        ax.set_title(f'Channel {i}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide unused subplots
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_rgb_interpretation(image):
    """Create an RGB interpretation of the rasterized image"""
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    num_channels = image.shape[0]
    
    if num_channels >= 3:
        # Use first 3 channels as RGB
        rgb_image = image[:3].transpose(1, 2, 0)
        # Normalize each channel
        for i in range(3):
            channel = rgb_image[:, :, i]
            rgb_image[:, :, i] = (channel - np.min(channel)) / (np.max(channel) - np.min(channel) + 1e-8)
    else:
        # Convert single/dual channel to grayscale
        if num_channels == 1:
            gray = image[0]
        else:
            gray = np.mean(image, axis=0)
        
        # Normalize
        gray = (gray - np.min(gray)) / (np.max(gray) - np.min(gray) + 1e-8)
        rgb_image = np.stack([gray, gray, gray], axis=-1)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(rgb_image, origin='lower')
    ax.set_title('RGB Interpretation')
    ax.axis('off')
    
    return fig


def print_batch_info(batch):
    """Print information about the batch"""
    print("\n" + "="*50)
    print("BATCH INFORMATION")
    print("="*50)
    
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"{key}: {value.shape} | dtype: {value.dtype} | min: {value.min().item():.3f} | max: {value.max().item():.3f}")
        elif isinstance(value, (list, tuple)):
            print(f"{key}: {type(value)} with {len(value)} elements")
        else:
            print(f"{key}: {type(value)} - {value}")


def visualize_trajectories(batch, sample_idx=0):
    """Visualize agent trajectories"""
    if "history_positions" in batch and "target_positions" in batch:
        history_pos = batch["history_positions"][sample_idx].cpu().numpy()  # [T_hist, 2]
        target_pos = batch["target_positions"][sample_idx].cpu().numpy()    # [T_fut, 2]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot history (blue)
        ax.plot(history_pos[:, 0], history_pos[:, 1], 'b-o', label='History', markersize=3)
        
        # Plot future target (red)
        ax.plot(target_pos[:, 0], target_pos[:, 1], 'r-o', label='Future Target', markersize=3)
        
        # Connect history to future
        if len(history_pos) > 0 and len(target_pos) > 0:
            ax.plot([history_pos[-1, 0], target_pos[0, 0]], 
                   [history_pos[-1, 1], target_pos[0, 1]], 'g--', alpha=0.5)
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Agent Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        return fig
    
    return None


def draw_vehicle_box(ax, pos, yaw, extent, color, alpha=0.8, zorder=10, linewidth=2):
    """Draw a vehicle as an oriented rectangular box"""
    # Ensure inputs are numpy arrays/scalars
    pos = np.asarray(pos).flatten()[:2]  # Ensure 2D position
    yaw = float(np.asarray(yaw).flatten()[0])  # Ensure scalar yaw
    extent = np.asarray(extent).flatten()[:2]  # Take only length and width
    
    # Vehicle dimensions
    length, width = extent[0], extent[1]
    
    # Skip if dimensions are invalid
    if length <= 0 or width <= 0:
        # Fallback to circle
        ax.scatter(pos[0], pos[1], color=color, s=100, alpha=alpha, 
                  zorder=zorder, edgecolors='white', linewidth=linewidth, marker='o')
        return
    
    # Create box corners in vehicle coordinate system (centered at origin)
    corners = np.array([
        [-length/2, -width/2],  # rear left
        [length/2, -width/2],   # front left  
        [length/2, width/2],    # front right
        [-length/2, width/2]    # rear right
    ])
    
    # Rotation matrix for vehicle heading
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])
    
    # Rotate corners and translate to vehicle position
    rotated_corners = corners @ rotation_matrix.T + pos.reshape(1, 2)
    
    # Create polygon patch
    vehicle_patch = patches.Polygon(rotated_corners, closed=True, 
                                   facecolor=color, edgecolor='white',
                                   alpha=alpha, zorder=zorder, linewidth=linewidth)
    ax.add_patch(vehicle_patch)
    
    # Add direction indicator (small line at front of vehicle)
    front_center = np.array([length/3, 0]) @ rotation_matrix.T + pos
    direction_end = np.array([length/2, 0]) @ rotation_matrix.T + pos
    ax.plot([front_center[0], direction_end[0]], [front_center[1], direction_end[1]], 
            color='white', linewidth=3, alpha=0.9, zorder=zorder+1)


def visualize_comprehensive_topdown(batch, sample_idx=0, title="nuPlan Trajectory Visualization"):
    """Create a single comprehensive top-down view matching official nuPlan style with vehicle boxes"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Debug: Print data information
    print(f"\n=== COORDINATE SYSTEM DEBUG FOR SAMPLE {sample_idx} ===")
    if "image" in batch:
        image_shape = batch["image"][sample_idx].shape
        print(f"Image tensor shape: {image_shape}")
        
    if "maps" in batch and batch["maps"] is not None:
        maps_shape = batch["maps"][sample_idx].shape  
        print(f"Maps tensor shape: {maps_shape}")
        
    # Detailed transformation matrix analysis
    if "raster_from_agent" in batch:
        raster_from_agent = batch["raster_from_agent"][sample_idx].cpu().numpy()
        print(f"Raster from agent transform:\n{raster_from_agent}")
        
        # Extract transformation components
        scale_x, scale_y = raster_from_agent[0, 0], raster_from_agent[1, 1]
        offset_x, offset_y = raster_from_agent[0, 2], raster_from_agent[1, 2]
        print(f"Scale: x={scale_x}, y={scale_y}")
        print(f"Offset: x={offset_x}, y={offset_y}")
        
        # Test transformation of key points
        test_points = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]]).T  # ego center, +x, +y, -x, -y
        raster_points = raster_from_agent @ test_points
        print(f"Agent->Raster coordinate transformation test:")
        for i, (agent_pt, raster_pt) in enumerate(zip(test_points.T, raster_points.T)):
            print(f"  Agent ({agent_pt[0]:+.1f}, {agent_pt[1]:+.1f}) -> Raster ({raster_pt[0]:+.1f}, {raster_pt[1]:+.1f})")
    
    if "agent_from_raster" in batch:
        agent_from_raster = batch["agent_from_raster"][sample_idx].cpu().numpy()
        print(f"Agent from raster transform:\n{agent_from_raster}")
    
    # Use official nuPlan visualization method with proper maps data
    rgb_idx_groups = [[0], [1], [2]]  # nuPlan mini uses 3 channels
    
    if batch["maps"] is None:
        # No map available, create black background
        _, h, w = batch["image"][sample_idx].shape[-2:]  # Get last 2 dimensions
        map_image = np.zeros((h, w, 3))
        print("No map data available, using black background")
    else:
        # Use the official RasterizedMap.to_img method with the correct maps tensor
        maps_tensor = batch["maps"][sample_idx]  # Use maps, not image
        if torch.is_tensor(maps_tensor):
            maps_tensor = maps_tensor.cpu()
        map_image = RasterizedMap.to_img(maps_tensor, rgb_idx_groups)
        print(f"Using official nuPlan visualization for {maps_tensor.shape[0]} map channels")
        print(f"Map image shape after RasterizedMap.to_img: {map_image.shape}")
        print(f"Map image value range: [{map_image.min():.3f}, {map_image.max():.3f}]")
        
        # Check map content at key locations
        h, w = map_image.shape[:2]
        center_x, center_y = w // 2, h // 2
        print(f"Map content check:")
        print(f"  Center pixel ({center_x}, {center_y}): {map_image[center_y, center_x]}")
        print(f"  Top-left pixel (0, 0): {map_image[0, 0]}")
        print(f"  Bottom-right pixel ({w-1}, {h-1}): {map_image[h-1, w-1]}")
    
    # Get coordinate transformation info
    raster_from_agent = batch["raster_from_agent"][sample_idx].cpu().numpy()
    pixel_size = 1.0 / 2.0  # 2 pixels per meter (from config)
    raster_size = map_image.shape[0]  # assume square
    ego_center = (-0.5, 0.0)  # from config
    
    # Calculate the coordinate range in meters based on raster configuration
    # The ego is at offset (-0.5, 0.0) which means ego is at 25% from left, 50% from bottom
    coord_range = (raster_size * pixel_size) / 2.0
    
    print(f"Coordinate system parameters:")
    print(f"  Coordinate range: ±{coord_range} meters")
    print(f"  Raster size: {raster_size}x{raster_size} pixels")
    print(f"  Pixel size: {pixel_size} m/px")
    print(f"  Ego center config: {ego_center}")
    
    # Check where ego should be in pixel coordinates
    ego_agent_pos = np.array([0, 0, 1])  # ego at origin in agent coords
    ego_raster_pos = raster_from_agent @ ego_agent_pos
    print(f"  Ego position in raster coordinates: ({ego_raster_pos[0]:.1f}, {ego_raster_pos[1]:.1f})")
    print(f"  Expected ego pixel: ({raster_size/2 + ego_center[0]*raster_size:.1f}, {raster_size/2 + ego_center[1]*raster_size:.1f})")
    
    # Fix: Calculate the proper extent based on the actual transformation
    # The transformation maps agent (0,0) to raster (56, 112)
    # So we need to adjust our world coordinate system accordingly
    ego_pixel_x, ego_pixel_y = ego_raster_pos[0], ego_raster_pos[1]
    
    # Convert raster pixel coordinates to world coordinates for extent
    # If ego is at pixel (56, 112), what world coordinates should this correspond to?
    # We want ego (agent coords 0,0) to be at world coords (0,0) in our plot
    world_min_x = -ego_pixel_x * pixel_size
    world_max_x = (raster_size - ego_pixel_x) * pixel_size
    world_min_y = -(raster_size - ego_pixel_y) * pixel_size  # Note: y-axis is flipped
    world_max_y = ego_pixel_y * pixel_size
    
    print(f"  Corrected extent: x=[{world_min_x:.1f}, {world_max_x:.1f}], y=[{world_min_y:.1f}, {world_max_y:.1f}]")
    
    # Display the official nuPlan map
    extent = [world_min_x, world_max_x, world_min_y, world_max_y]
    ax.imshow(map_image, origin='lower', extent=extent, interpolation='nearest')
    
    # Define colors matching nuPlan style
    colors = {
        'ego': '#FF0000',           # Red for ego vehicle
        'same_lane': '#0066FF',     # Blue for same lane traffic  
        'target_lane': '#00FF00',   # Green for target lane traffic
        'other_lane': '#FF00FF',    # Magenta/Purple for other lane traffic
        'general_other': '#888888'  # Gray for general other vehicles
    }
    
    # Plot ego vehicle trajectory and box
    ego_pos = None
    ego_yaw = None
    if "history_positions" in batch and "target_positions" in batch:
        history_pos = batch["history_positions"][sample_idx].cpu().numpy()
        target_pos = batch["target_positions"][sample_idx].cpu().numpy()
        
        print(f"\nEgo trajectory analysis:")
        print(f"  History positions shape: {history_pos.shape}")
        print(f"  Target positions shape: {target_pos.shape}")
        if len(history_pos) > 0:
            print(f"  Ego current position (agent coords): [{history_pos[-1, 0]:.2f}, {history_pos[-1, 1]:.2f}]")
            # Convert to raster coordinates for verification
            ego_agent_pos = np.array([history_pos[-1, 0], history_pos[-1, 1], 1])
            ego_raster_pos = raster_from_agent @ ego_agent_pos
            print(f"  Ego current position (raster coords): [{ego_raster_pos[0]:.2f}, {ego_raster_pos[1]:.2f}]")
        
        # Plot ego history trajectory (lighter line)
        if len(history_pos) > 0:
            ax.plot(history_pos[:, 0], history_pos[:, 1], '-', 
                   color=colors['ego'], linewidth=2, alpha=0.5, zorder=8)
            ego_pos = history_pos[-1]
        
        # Plot ego future target trajectory (dashed line)
        if len(target_pos) > 0:
            ax.plot(target_pos[:, 0], target_pos[:, 1], '--', 
                   color=colors['ego'], linewidth=2, alpha=0.6, zorder=8)
            
            # Connect history to future
            if len(history_pos) > 0:
                ax.plot([history_pos[-1, 0], target_pos[0, 0]], 
                       [history_pos[-1, 1], target_pos[0, 1]], 
                       '--', color=colors['ego'], linewidth=2, alpha=0.6, zorder=8)
    
    # Draw ego vehicle box
    if "history_yaws" in batch and "extent" in batch and ego_pos is not None:
        ego_yaw = batch["history_yaws"][sample_idx, -1].cpu().numpy()  # Shape: (1,)
        ego_extent = batch["extent"][sample_idx].cpu().numpy()  # Shape: (3,)
        print(f"  Ego yaw: {float(ego_yaw):.3f} rad ({np.degrees(float(ego_yaw)):.1f} deg)")
        print(f"  Ego extent: {ego_extent}")
        
        draw_vehicle_box(ax, ego_pos, ego_yaw, ego_extent, colors['ego'], 
                        alpha=0.9, zorder=15, linewidth=3)
        
        # Add label for legend
        ax.plot([], [], 's', color=colors['ego'], markersize=8, label='Ego Vehicle')
    
    # Plot other agents with boxes
    if "all_other_agents_history_positions" in batch:
        other_agents_hist = batch["all_other_agents_history_positions"][sample_idx]
        if torch.is_tensor(other_agents_hist):
            other_agents_hist = other_agents_hist.cpu().numpy()
        
        print(f"\nOther agents analysis:")
        print(f"  Other agents history shape: {other_agents_hist.shape}")
        
        # Get other agent data
        other_agents_yaw = None
        other_agents_extent = None
        if "all_other_agents_history_yaws" in batch:
            other_agents_yaw = batch["all_other_agents_history_yaws"][sample_idx].cpu().numpy()  # Shape: (47, 31, 1)
        if "all_other_agents_extents" in batch:
            other_agents_extent = batch["all_other_agents_extents"][sample_idx].cpu().numpy()  # Shape: (47, 3)
        
        # Track which legend entries we've added
        legend_added = {'same_lane': False, 'target_lane': False, 'other_lane': False}
        
        agent_count = 0
        for agent_idx in range(min(5, other_agents_hist.shape[0])):  # Limit to 5 for detailed debugging
            agent_traj = other_agents_hist[agent_idx]
            # Filter out invalid positions
            valid_mask = ~((agent_traj == 0).all(axis=1) | (np.abs(agent_traj) > coord_range).any(axis=1))
            
            if valid_mask.sum() > 1:  # Need at least 2 points
                valid_traj = agent_traj[valid_mask]
                agent_final_pos = valid_traj[-1]
                
                print(f"  Agent {agent_idx}:")
                print(f"    Final position (agent coords): [{agent_final_pos[0]:.2f}, {agent_final_pos[1]:.2f}]")
                
                # Convert to raster coordinates for verification
                agent_pos_homo = np.array([agent_final_pos[0], agent_final_pos[1], 1])
                agent_raster_pos = raster_from_agent @ agent_pos_homo
                print(f"    Final position (raster coords): [{agent_raster_pos[0]:.2f}, {agent_raster_pos[1]:.2f}]")
                
                # Check what's at this location on the map
                raster_x_px = int(agent_raster_pos[0])
                raster_y_px = int(agent_raster_pos[1])
                if 0 <= raster_x_px < map_image.shape[1] and 0 <= raster_y_px < map_image.shape[0]:
                    map_value = map_image[raster_y_px, raster_x_px]
                    print(f"    Map value at position: {map_value}")
                else:
                    print(f"    Position is outside map bounds")
                
                # Determine vehicle type based on position relative to ego
                ego_pos_ref = ego_pos if ego_pos is not None else np.array([0, 0])
                
                # Simple lane classification based on lateral distance
                lateral_dist = abs(agent_final_pos[1] - ego_pos_ref[1])
                longitudinal_dist = agent_final_pos[0] - ego_pos_ref[0]
                
                if lateral_dist < 2.0:  # Same lane (within 2m laterally)
                    color = colors['same_lane']
                    label = 'Same Lane Traffic' if not legend_added['same_lane'] else None
                    legend_added['same_lane'] = True
                elif lateral_dist < 6.0 and longitudinal_dist > 0:  # Target lane ahead
                    color = colors['target_lane'] 
                    label = 'Target Lane Traffic' if not legend_added['target_lane'] else None
                    legend_added['target_lane'] = True
                else:  # Other lane
                    color = colors['other_lane']
                    label = 'Other Lane Traffic' if not legend_added['other_lane'] else None
                    legend_added['other_lane'] = True
                
                # Plot agent trajectory (lighter line)
                ax.plot(valid_traj[:, 0], valid_traj[:, 1], '-', 
                       color=color, linewidth=1.5, alpha=0.4, zorder=7)
                
                # Draw vehicle box at final position
                if (other_agents_yaw is not None and other_agents_extent is not None and 
                    agent_idx < len(other_agents_yaw) and agent_idx < len(other_agents_extent)):
                    
                    # Get the final valid yaw and extent - fix shape handling
                    if valid_mask.any():
                        # Find the last valid index
                        valid_indices = np.where(valid_mask)[0]
                        last_valid_idx = valid_indices[-1] if len(valid_indices) > 0 else -1
                        agent_yaw = other_agents_yaw[agent_idx, last_valid_idx]  # Shape: (1,)
                    else:
                        agent_yaw = other_agents_yaw[agent_idx, -1]  # Shape: (1,)
                    
                    agent_extent = other_agents_extent[agent_idx]  # Shape: (3,)
                    
                    # Check if extent is valid (not all zeros)
                    if not np.allclose(agent_extent[:2], 0):  # Check only length and width
                        draw_vehicle_box(ax, agent_final_pos, agent_yaw, agent_extent, 
                                       color, alpha=0.8, zorder=12, linewidth=2)
                    else:
                        # Fallback to circle if no extent data
                        ax.scatter(agent_final_pos[0], agent_final_pos[1], 
                                 color=color, s=100, alpha=0.8, zorder=12, 
                                 edgecolors='white', linewidth=1, marker='o')
                else:
                    # Fallback to circle if no yaw/extent data
                    ax.scatter(agent_final_pos[0], agent_final_pos[1], 
                             color=color, s=100, alpha=0.8, zorder=12, 
                             edgecolors='white', linewidth=1, marker='o')
                
                # Add to legend (using square marker to represent vehicles)
                if label is not None:
                    ax.plot([], [], 's', color=color, markersize=6, label=label)
                
                agent_count += 1
        
        # Continue with remaining agents (less detailed output)
        for agent_idx in range(5, min(10, other_agents_hist.shape[0])):
            agent_traj = other_agents_hist[agent_idx]
            valid_mask = ~((agent_traj == 0).all(axis=1) | (np.abs(agent_traj) > coord_range).any(axis=1))
            
            if valid_mask.sum() > 1:
                valid_traj = agent_traj[valid_mask]
                agent_final_pos = valid_traj[-1]
                
                print(f"  Agent {agent_idx}: final position [{agent_final_pos[0]:.2f}, {agent_final_pos[1]:.2f}]")
                
                # Simple classification and drawing (same as before)
                ego_pos_ref = ego_pos if ego_pos is not None else np.array([0, 0])
                lateral_dist = abs(agent_final_pos[1] - ego_pos_ref[1])
                longitudinal_dist = agent_final_pos[0] - ego_pos_ref[0]
                
                if lateral_dist < 2.0:
                    color = colors['same_lane']
                    label = 'Same Lane Traffic' if not legend_added['same_lane'] else None
                    legend_added['same_lane'] = True
                elif lateral_dist < 6.0 and longitudinal_dist > 0:
                    color = colors['target_lane'] 
                    label = 'Target Lane Traffic' if not legend_added['target_lane'] else None
                    legend_added['target_lane'] = True
                else:
                    color = colors['other_lane']
                    label = 'Other Lane Traffic' if not legend_added['other_lane'] else None
                    legend_added['other_lane'] = True
                
                ax.plot(valid_traj[:, 0], valid_traj[:, 1], '-', 
                       color=color, linewidth=1.5, alpha=0.4, zorder=7)
                
                if (other_agents_yaw is not None and other_agents_extent is not None and 
                    agent_idx < len(other_agents_yaw) and agent_idx < len(other_agents_extent)):
                    
                    if valid_mask.any():
                        valid_indices = np.where(valid_mask)[0]
                        last_valid_idx = valid_indices[-1] if len(valid_indices) > 0 else -1
                        agent_yaw = other_agents_yaw[agent_idx, last_valid_idx]
                    else:
                        agent_yaw = other_agents_yaw[agent_idx, -1]
                    
                    agent_extent = other_agents_extent[agent_idx]
                    
                    if not np.allclose(agent_extent[:2], 0):
                        draw_vehicle_box(ax, agent_final_pos, agent_yaw, agent_extent, 
                                       color, alpha=0.8, zorder=12, linewidth=2)
                    else:
                        ax.scatter(agent_final_pos[0], agent_final_pos[1], 
                                 color=color, s=100, alpha=0.8, zorder=12, 
                                 edgecolors='white', linewidth=1, marker='o')
                else:
                    ax.scatter(agent_final_pos[0], agent_final_pos[1], 
                             color=color, s=100, alpha=0.8, zorder=12, 
                             edgecolors='white', linewidth=1, marker='o')
                
                if label is not None:
                    ax.plot([], [], 's', color=color, markersize=6, label=label)
    
    print("=== END COORDINATE SYSTEM DEBUG ===\n")
    
    # Formatting to match nuPlan style
    ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold', color='white')
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=20)
    
    # Style the legend to match nuPlan
    legend = ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Grid styling
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    ax.set_aspect('equal')
    
    # Set limits to match the corrected coordinate system
    ax.set_xlim(world_min_x, world_max_x)
    ax.set_ylim(world_min_y, world_max_y)
    
    # Style tick labels
    ax.tick_params(colors='white', labelsize=12)
    
    # Add scenario info box
    scenario_info = f'Sample {sample_idx} | Frame 1/1'
    ax.text(0.02, 0.98, scenario_info, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', edgecolor='white', alpha=0.8),
            color='white')
    
    # Add map info
    if "image" in batch:
        image_shape = batch["image"][sample_idx].shape
        map_info = f'{image_shape[0]} channels | {image_shape[1]}×{image_shape[2]}px | {pixel_size:.1f}m/px'
        ax.text(0.02, 0.02, map_info, transform=ax.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='black', edgecolor='white', alpha=0.8),
                color='white')
    
    plt.tight_layout()
    return fig


def main():
    print("Getting configuration...")
    config = get_registered_experiment_config("trajdata_nuplan_bc")
    
    # Update data path
    config.train.trajdata_data_dirs = {
        "nuplan_mini": "/home/stud/nguyenti/storage/user/datasets/nuplan/"
    }
    
    print("Setting up trajdata environment...")
    setup_trajdata_env(config)
    
    print("Creating dataloader...")
    try:
        train_loader, config = create_dataloader(config)
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return
    
    print("Loading a batch of data...")
    try:
        # Get first batch
        batch_iter = iter(train_loader)
        print("Created batch iterator successfully")
        
        batch = next(batch_iter)
        print(f"Got batch, type: {type(batch)}")
        
        if batch is None:
            print("ERROR: Batch is None!")
            return
        
        # Parse batch
        print("Parsing batch...")
        batch = batch_utils().parse_batch(batch)
        print(f"Parsed batch, type: {type(batch)}")
        
        if batch is None:
            print("ERROR: Parsed batch is None!")
            return
            
    except Exception as e:
        print(f"Error loading batch: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print batch information
    print_batch_info(batch)
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Visualize first sample in batch
    sample_idx = 0
    
    if "image" in batch:
        print(f"\nVisualizing rasterized image for sample {sample_idx}...")
        image = batch["image"][sample_idx]  # [C, H, W]
        
        # Create visualizations
        fig1 = visualize_raster_channels(image, f"Rasterized Input - Sample {sample_idx}")
        fig2 = visualize_rgb_interpretation(image)
        
        # Save visualizations
        fig1.savefig(f"visualizations/raster_channels_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        fig2.savefig(f"visualizations/rgb_interpretation_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        
        print(f"Saved raster channels visualization: visualizations/raster_channels_sample_{sample_idx}.png")
        print(f"Saved RGB interpretation: visualizations/rgb_interpretation_sample_{sample_idx}.png")
        
        # Close figures to free memory
        plt.close(fig1)
        plt.close(fig2)
    
    # Visualize trajectories if available
    traj_fig = visualize_trajectories(batch, sample_idx)
    if traj_fig is not None:
        traj_fig.savefig(f"visualizations/trajectory_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        print(f"Saved trajectory visualization: visualizations/trajectory_sample_{sample_idx}.png")
        plt.close(traj_fig)
    
    # Let's also check a few more samples
    print("\nChecking multiple samples...")
    for i in range(min(3, len(batch["image"]))):
        if "image" in batch:
            image = batch["image"][i]
            print(f"Sample {i}: Image shape {image.shape}, min: {image.min().item():.3f}, max: {image.max().item():.3f}")
            
            # Create comprehensive top-down view for each sample
            comprehensive_fig = visualize_comprehensive_topdown(batch, i, f"Comprehensive Top-Down View - Sample {i}")
            comprehensive_fig.savefig(f"visualizations/comprehensive_sample_{i}.png", dpi=150, bbox_inches='tight')
            print(f"Saved comprehensive top-down view for sample {i}: visualizations/comprehensive_sample_{i}.png")
            plt.close(comprehensive_fig)
            
            # Quick channel visualization
            fig = visualize_raster_channels(image, f"Sample {i}")
            fig.savefig(f"visualizations/sample_{i}_channels.png", dpi=150, bbox_inches='tight')
            print(f"Saved sample {i} channels: visualizations/sample_{i}_channels.png")
            plt.close(fig)  # Close to save memory
    
    print("\nVisualization complete! Check the 'visualizations/' directory for saved images.")


if __name__ == "__main__":
    main() 