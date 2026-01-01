import cv2
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Analyze difference between two images.")
    parser.add_argument("image1", help="Path to the first image")
    parser.add_argument("image2", help="Path to the second image")
    parser.add_argument("--output", default="diff_result.png", help="Path to save the difference image")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image1):
        print(f"Error: File not found: {args.image1}")
        sys.exit(1)
    if not os.path.exists(args.image2):
        print(f"Error: File not found: {args.image2}")
        sys.exit(1)
        
    img1 = cv2.imread(args.image1)
    img2 = cv2.imread(args.image2)
    
    if img1 is None:
        print(f"Error: Could not read image1 at {args.image1}")
        sys.exit(1)
    if img2 is None:
        print(f"Error: Could not read image2 at {args.image2}")
        sys.exit(1)
        
    if img1.shape != img2.shape:
        print(f"Error: Images have different shapes: {img1.shape} vs {img2.shape}")
        sys.exit(1)
        
    # Calculate absolute difference
    diff = cv2.absdiff(img1, img2)
    
    # Save the raw difference
    cv2.imwrite(args.output, diff)
    print(f"Difference image saved to {args.output}")
    
    # Calculate statistics
    mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    print(f"Global Stats:")
    print(f"  MSE: {mse:.4f}")
    print(f"  Max Difference: {np.max(diff)}")
    print(f"  Mean Difference: {np.mean(diff):.4f}")

    # Detailed analysis on non-zero differences
    # If color image, convert to grayscale for magnitude analysis, or analyze per channel
    # Here we look at the max difference across channels per pixel for detection
    if len(diff.shape) == 3:
        diff_max_per_pixel = np.max(diff, axis=2)
    else:
        diff_max_per_pixel = diff

    non_zero_mask = diff_max_per_pixel > 0
    num_diff_pixels = np.count_nonzero(non_zero_mask)
    total_pixels = diff_max_per_pixel.size
    
    if num_diff_pixels > 0:
        # Get values where there is a difference
        diff_values = diff_max_per_pixel[non_zero_mask]
        
        print("\n--- Detailed Difference Analysis (Non-zero pixels) ---")
        print(f"Different Pixels: {num_diff_pixels} / {total_pixels} ({num_diff_pixels/total_pixels*100:.2f}%)")
        print(f"Min Diff (Non-zero): {np.min(diff_values)}")
        print(f"Max Diff (Non-zero): {np.max(diff_values)}")
        print(f"Mean Diff (Non-zero): {np.mean(diff_values):.2f}")
        print(f"Median Diff (Non-zero): {np.median(diff_values):.2f}")
        print(f"Std Dev (Non-zero): {np.std(diff_values):.2f}")
        
        print("\nPercentiles (of difference magnitude):")
        for p in [50, 75, 90, 95, 99]:
            print(f"  {p}th percentile: {np.percentile(diff_values, p):.2f}")
            
        if len(diff.shape) == 3:
            print("\nPer-channel Stats (Max/Mean):")
            b, g, r = cv2.split(diff)
            print(f"  Blue:  Max={np.max(b)}, Mean={np.mean(b):.4f}")
            print(f"  Green: Max={np.max(g)}, Mean={np.mean(g):.4f}")
            print(f"  Red:   Max={np.max(r)}, Mean={np.mean(r):.4f}")
    else:
        print("\nImages are identical.")

    # Visualization: Pixel Difference Plot
    plt.figure(figsize=(12, 6))
    
    # Use max difference across channels for each pixel if color
    if len(diff.shape) == 3:
        # (H, W) array of max diffs
        plot_data = np.max(diff, axis=2).flatten()
    else:
        plot_data = diff.flatten()
        
    # X-axis: Pixel Index (0 to Total Pixels)
    # Y-axis: Difference Value
    x_axis = range(len(plot_data))
    
    plt.plot(x_axis, plot_data, color='blue', linewidth=0.5, alpha=0.7)
    
    plt.title('Difference per Pixel (Flattened)')
    plt.xlabel(f'Pixel Index (0 - {len(plot_data)-1})')
    plt.ylabel('Difference Value (0-255)')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = args.output.replace(".png", "_plot.png")
    plt.savefig(plot_path)
    print(f"\nPixel difference plot saved to {plot_path}")
    plt.close()

    # Visualization: Sorted Pixel Difference Plot (More intuitive for "how much is black")
    plt.figure(figsize=(12, 6))
    
    # Sort the difference values to show distribution shape
    sorted_plot_data = np.sort(plot_data)
    
    plt.plot(x_axis, sorted_plot_data, color='red', linewidth=1, alpha=0.8)
    
    # Mark the transition point if possible (e.g. where diff > 5)
    vis_threshold = 5
    num_black = np.sum(sorted_plot_data <= vis_threshold)
    pct_black = num_black / len(sorted_plot_data) * 100
    
    plt.axvline(x=num_black, color='green', linestyle='--', alpha=0.5, label=f'Diff <= {vis_threshold} ({pct_black:.1f}%)')
    plt.legend()
    
    plt.title(f'Sorted Difference Distribution ({pct_black:.1f}% pixels are near-black)')
    plt.xlabel('Pixel Count (Sorted by Difference)')
    plt.ylabel('Difference Value (0-255)')
    plt.grid(True, alpha=0.3)
    
    # Save the sorted plot
    sorted_plot_path = args.output.replace(".png", "_sorted_plot.png")
    plt.savefig(sorted_plot_path)
    print(f"Sorted pixel difference plot saved to {sorted_plot_path}")
    plt.close()

    # Visualization: Significant Difference Plot (Sorted, > 5)
    # Ensure diff_values is defined
    if 'diff_values' not in locals():
         diff_values = np.array([])
         
    # Filter for values > 5
    sig_diff_values = diff_values[diff_values > 5]
    
    if len(sig_diff_values) > 0:
        plt.figure(figsize=(12, 6))
        
        # Sort values for better visualization
        sorted_diff_values = np.sort(sig_diff_values)
        x_axis_nonzero = range(len(sorted_diff_values))
        
        plt.plot(x_axis_nonzero, sorted_diff_values, color='orange', linewidth=1, alpha=0.8)
        
        plt.title(f'Significant Difference Distribution (Sorted, > 5, {len(sig_diff_values)} pixels)')
        plt.xlabel('Pixel Count (Sorted by Difference)')
        plt.ylabel('Difference Value (>5)')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        nonzero_plot_path = args.output.replace(".png", "_significant_sorted_plot.png")
        plt.savefig(nonzero_plot_path)
        print(f"Significant difference sorted plot saved to {nonzero_plot_path}")
        plt.close()

if __name__ == "__main__":
    main()

