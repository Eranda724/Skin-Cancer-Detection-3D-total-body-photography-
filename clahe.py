#!/usr/bin/env python
## @file
# @title Contrast Limited Adaptive Histogram Equalization - Batch Processor
# @brief Process entire dataset using CLAHE with multiple configurations
# @author Modified for batch processing
# @date 2025
# @version 1.0
# @licence GPLv3

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
import glob
from pathlib import Path

def warn(text):
    print('\033[38;5;226m' + text + '\033[0m')

def error(text, errorcode):
    print('\033[38;5;196m' + text + '\033[0m')
    exit(errorcode)

def info(text):
    print('\033[38;5;82m' + text + '\033[0m')

def create_output_directories(base_output_dir):
    """Create output directories for different CLAHE configurations"""
    directories = {
        'default': os.path.join(base_output_dir, 'default'),
        'light': os.path.join(base_output_dir, 'light'),
        'strong': os.path.join(base_output_dir, 'strong'),
        'input_copies': os.path.join(base_output_dir, 'input_copies')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def apply_clahe(image, clip_limit=2.0, tile_size=(8, 8)):
    """Apply CLAHE to an image"""
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    # Convert color space from RGB to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split into channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel
    l = clahe.apply(l)
    
    # Merge channels
    lab = cv2.merge((l, a, b))
    
    # Convert back to RGB
    output = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return output

def process_single_image(input_path, output_dirs, image_number, clahe_configs):
    """Process a single image with multiple CLAHE configurations"""
    try:
        # Read image
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if image is None:
            warn(f"Could not read image: {input_path}")
            return False
        
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save original image copy
        input_filename = f"{image_number}.jpg"
        input_save_path = os.path.join(output_dirs['input_copies'], input_filename)
        cv2.imwrite(input_save_path, image)  # Save in original BGR format
        
        # Process with different CLAHE configurations
        for config_name, config in clahe_configs.items():
            # Apply CLAHE
            processed_image = apply_clahe(
                image_rgb, 
                clip_limit=config['clip_limit'], 
                tile_size=config['tile_size']
            )
            
            # Convert back to BGR for saving
            processed_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            
            # Save processed image
            output_filename = f"{image_number}.jpg"
            output_path = os.path.join(output_dirs[config_name], output_filename)
            cv2.imwrite(output_path, processed_bgr)
        
        info(f"Processed image {image_number}: {os.path.basename(input_path)}")
        return True
        
    except Exception as e:
        error(f"Error processing {input_path}: {str(e)}", 1)
        return False

def create_readme_images(output_dirs, clahe_configs):
    """Create README demonstration images using the first processed image"""
    try:
        # Find the first image in input_copies
        input_files = glob.glob(os.path.join(output_dirs['input_copies'], "*.jpg"))
        if not input_files:
            warn("No input images found for README generation")
            return
        
        # Use the first image (should be 1.jpg)
        first_input = input_files[0]
        input_image = cv2.imread(first_input)
        input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Create README images
        readme_base = os.path.dirname(output_dirs['default'])  # Go back to main output directory
        
        # Save input image for README
        readme_input_path = os.path.join(readme_base, "readme-input.jpg")
        cv2.imwrite(readme_input_path, input_image)
        
        # Generate and save CLAHE variants for README
        for config_name, config in clahe_configs.items():
            processed_image = apply_clahe(
                input_rgb,
                clip_limit=config['clip_limit'],
                tile_size=config['tile_size']
            )
            processed_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            
            readme_filename = f"readme-clahe-{config_name}.jpg"
            readme_path = os.path.join(readme_base, readme_filename)
            cv2.imwrite(readme_path, processed_bgr)
        
        info("README images created successfully")
        
    except Exception as e:
        warn(f"Could not create README images: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description='Batch process images using CLAHE algorithm with multiple configurations.'
    )
    parser.add_argument(
        '-i', '--input', 
        dest='input_dir', 
        type=str, 
        default='images', 
        help='Input directory containing images'
    )
    parser.add_argument(
        '-o', '--output', 
        dest='output_dir', 
        type=str, 
        default='output', 
        help='Output directory for processed images'
    )
    parser.add_argument(
        '-si', '--show-images', 
        dest='show_images', 
        action='store_true', 
        help='Show comparison images'
    )
    parser.add_argument(
        '--extensions',
        dest='extensions',
        nargs='+',
        default=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help='Image file extensions to process'
    )
    
    args = parser.parse_args()
    
    # Configuration
    input_dir = args.input_dir
    output_dir = args.output_dir
    show_images = args.show_images
    extensions = args.extensions
    
    # CLAHE configurations
    clahe_configs = {
        'default': {'clip_limit': 2.0, 'tile_size': (8, 8)},
        'light': {'clip_limit': 1.0, 'tile_size': (8, 8)},
        'strong': {'clip_limit': 4.0, 'tile_size': (8, 8)}
    }
    
    # Check input directory
    if not os.path.exists(input_dir):
        error(f"Input directory does not exist: {input_dir}", 1)
    
    # Create output directories
    output_dirs = create_output_directories(output_dir)
    info(f"Created output directories in: {output_dir}")
    
    # Find all image files
    image_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*.{ext}")
        image_files.extend(glob.glob(pattern, recursive=False))
        pattern = os.path.join(input_dir, f"*.{ext.upper()}")
        image_files.extend(glob.glob(pattern, recursive=False))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        error(f"No image files found in {input_dir} with extensions {extensions}", 1)
    
    info(f"Found {len(image_files)} images to process")
    
    # Process images
    successful_count = 0
    for i, image_path in enumerate(image_files, 1):
        if process_single_image(image_path, output_dirs, i, clahe_configs):
            successful_count += 1
    
    info(f"Successfully processed {successful_count}/{len(image_files)} images")
    
    # Create README demonstration images
    create_readme_images(output_dirs, clahe_configs)
    
    # Show comparison if requested
    if show_images and successful_count > 0:
        try:
            # Show comparison for first image
            first_input = os.path.join(output_dirs['input_copies'], "1.jpg")
            first_default = os.path.join(output_dirs['default'], "1.jpg")
            first_light = os.path.join(output_dirs['light'], "1.jpg")
            first_strong = os.path.join(output_dirs['strong'], "1.jpg")
            
            if all(os.path.exists(f) for f in [first_input, first_default, first_light, first_strong]):
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle('CLAHE Comparison - First Image', fontsize=16)
                
                # Load and display images
                images = [
                    (cv2.cvtColor(cv2.imread(first_input), cv2.COLOR_BGR2RGB), "Original"),
                    (cv2.cvtColor(cv2.imread(first_default), cv2.COLOR_BGR2RGB), "Default (2.0)"),
                    (cv2.cvtColor(cv2.imread(first_light), cv2.COLOR_BGR2RGB), "Light (1.0)"),
                    (cv2.cvtColor(cv2.imread(first_strong), cv2.COLOR_BGR2RGB), "Strong (4.0)")
                ]
                
                for idx, (img, title) in enumerate(images):
                    row, col = idx // 2, idx % 2
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(title)
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                
                plt.tight_layout()
                plt.show()
            
        except Exception as e:
            warn(f"Could not display comparison images: {str(e)}")
    
    info("Batch processing completed!")

if __name__ == '__main__':
    main()