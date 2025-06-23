#!/usr/bin/env python3
"""
Simple script to visualize trajdata rasterized images.
This is a lightweight version for quick visualization during training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Set matplotlib to non-interactive backend to avoid display issues
plt.switch_backend('Agg')


def visualize_batch_images(batch, max_samples=3, save_dir="visualizations"):
    """
    Visualize images from a training batch
    
    Args:
        batch: Training batch dictionary containing 'image' key
        max_samples: Maximum number of samples to visualize
        save_dir: Directory to save visualizations
    """
    if "image" not in batch:
        print("No 'image' key found in batch")
        return
    
    images = batch["image"]  # Shape: [B, C, H, W]
    batch_size, num_channels, height, width = images.shape
    
    print(f"Batch image shape: {images.shape}")
    print(f"Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize up to max_samples
    num_samples = min(max_samples, batch_size)
    
    for sample_idx in range(num_samples):
        image = images[sample_idx]  # [C, H, W]
        
        # Create figure for this sample
        fig, axes = plt.subplots(2, (num_channels + 1) // 2, figsize=(15, 8))
        fig.suptitle(f'Sample {sample_idx} - Rasterized Input ({num_channels} channels)', fontsize=14)
        
        if num_channels == 1:
            axes = [axes] if not isinstance(axes, np.ndarray) else axes.flatten()
        else:
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        # Plot each channel
        for ch in range(num_channels):
            ax = axes[ch] if len(axes) > 1 else axes[0]
            
            # Get channel data
            channel_data = image[ch].cpu().numpy()
            
            # Normalize for visualization
            ch_min, ch_max = channel_data.min(), channel_data.max()
            if ch_max > ch_min:
                channel_norm = (channel_data - ch_min) / (ch_max - ch_min)
            else:
                channel_norm = channel_data
            
            # Plot
            im = ax.imshow(channel_norm, cmap='viridis', origin='lower')
            ax.set_title(f'Channel {ch}\nRange: [{ch_min:.2f}, {ch_max:.2f}]')
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for ch in range(num_channels, len(axes)):
            axes[ch].axis('off')
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(save_dir, f"sample_{sample_idx}_channels.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        # Create RGB interpretation if we have multiple channels
        if num_channels >= 3:
            fig_rgb, ax_rgb = plt.subplots(1, 1, figsize=(8, 8))
            
            # Use first 3 channels as RGB
            rgb_data = image[:3].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            
            # Normalize each channel independently
            for i in range(3):
                ch_data = rgb_data[:, :, i]
                ch_min, ch_max = ch_data.min(), ch_data.max()
                if ch_max > ch_min:
                    rgb_data[:, :, i] = (ch_data - ch_min) / (ch_max - ch_min)
            
            ax_rgb.imshow(rgb_data, origin='lower')
            ax_rgb.set_title(f'Sample {sample_idx} - RGB Interpretation (Ch 0-2)')
            ax_rgb.axis('off')
            
            # Save RGB version
            rgb_path = os.path.join(save_dir, f"sample_{sample_idx}_rgb.png")
            plt.savefig(rgb_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {rgb_path}")
            
            plt.close(fig_rgb)
        
        plt.close(fig)


def visualize_trajectories(batch, max_samples=3, save_dir="visualizations"):
    """
    Visualize agent trajectories from a batch
    
    Args:
        batch: Training batch dictionary
        max_samples: Maximum number of samples to visualize
        save_dir: Directory to save visualizations
    """
    if "history_positions" not in batch or "target_positions" not in batch:
        print("No trajectory data found in batch")
        return
    
    history_pos = batch["history_positions"]  # [B, T_hist, 2]
    target_pos = batch["target_positions"]    # [B, T_fut, 2]
    
    batch_size = history_pos.shape[0]
    num_samples = min(max_samples, batch_size)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for sample_idx in range(num_samples):
        hist = history_pos[sample_idx].cpu().numpy()  # [T_hist, 2]
        target = target_pos[sample_idx].cpu().numpy()  # [T_fut, 2]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot history trajectory
        ax.plot(hist[:, 0], hist[:, 1], 'b-o', label='History', markersize=4, linewidth=2)
        
        # Plot target trajectory
        ax.plot(target[:, 0], target[:, 1], 'r-o', label='Target Future', markersize=4, linewidth=2)
        
        # Connect last history point to first target point
        if len(hist) > 0 and len(target) > 0:
            ax.plot([hist[-1, 0], target[0, 0]], 
                   [hist[-1, 1], target[0, 1]], 'g--', alpha=0.7, linewidth=2)
        
        # Mark start and end points
        if len(hist) > 0:
            ax.plot(hist[0, 0], hist[0, 1], 'go', markersize=8, label='Start')
        if len(target) > 0:
            ax.plot(target[-1, 0], target[-1, 1], 'ro', markersize=8, label='End')
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'Sample {sample_idx} - Agent Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Save
        traj_path = os.path.join(save_dir, f"sample_{sample_idx}_trajectory.png")
        plt.savefig(traj_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {traj_path}")
        
        plt.close(fig)


def print_batch_summary(batch):
    """Print a summary of the batch contents"""
    print("\n" + "="*60)
    print("BATCH SUMMARY")
    print("="*60)
    
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"{key:20s}: {str(value.shape):20s} | {str(value.dtype):10s} | [{value.min().item():8.3f}, {value.max().item():8.3f}]")
        elif isinstance(value, (list, tuple)):
            print(f"{key:20s}: {str(type(value)):20s} | Length: {len(value)}")
        else:
            print(f"{key:20s}: {str(type(value)):20s} | {str(value)}")
    
    print("="*60)


# Example usage function that can be called from training scripts
def quick_visualize(batch, sample_idx=0, save_dir="quick_viz"):
    """
    Quick visualization function that can be called during training
    
    Args:
        batch: Training batch
        sample_idx: Index of sample to visualize
        save_dir: Directory to save images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Print basic info
    print(f"\nQuick visualization of sample {sample_idx}")
    
    # Visualize image if available
    if "image" in batch:
        image = batch["image"][sample_idx]  # [C, H, W]
        num_channels = image.shape[0]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Show first channel or RGB interpretation
        if num_channels >= 3:
            # RGB interpretation
            rgb_data = image[:3].cpu().numpy().transpose(1, 2, 0)
            for i in range(3):
                ch = rgb_data[:, :, i]
                rgb_data[:, :, i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
            ax.imshow(rgb_data, origin='lower')
            ax.set_title(f'Sample {sample_idx} - RGB View')
        else:
            # Single channel
            ch_data = image[0].cpu().numpy()
            ax.imshow(ch_data, cmap='viridis', origin='lower')
            ax.set_title(f'Sample {sample_idx} - Channel 0')
        
        ax.axis('off')
        
        save_path = os.path.join(save_dir, f"quick_sample_{sample_idx}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved quick visualization: {save_path}")


if __name__ == "__main__":
    print("This is a utility module for visualizing trajdata images.")
    print("Import and use the functions in your training scripts.")
    print("\nExample usage:")
    print("  from simple_visualize_trajdata import visualize_batch_images, print_batch_summary")
    print("  print_batch_summary(batch)")
    print("  visualize_batch_images(batch, max_samples=2)") 