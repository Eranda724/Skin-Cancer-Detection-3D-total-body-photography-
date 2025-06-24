########################################
# FOR USAGE: RUN BELOW COMMAND         #
# python enhanced_scan.py -i images/m1.jpg #
########################################

import numpy as np
import argparse
import cv2
import imutils
import os
from pathlib import Path

def four_point_transform(image, pts):
    """
    Transform the image to a top-down view using four corner points
    """
    # Order the points in the order: top-left, top-right, bottom-right, bottom-left
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Construct the set of destination points to obtain a "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def order_points(pts):
    """
    Order points in the order: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    # Top-left point has the smallest sum
    # Bottom-right point has the largest sum
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right point has the smallest difference
    # Bottom-left point has the largest difference
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def create_directories():
    """Create necessary directories if they don't exist"""
    Path("images").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    print("Created 'images' and 'output' directories")

def enhance_image_preprocessing(image):
    """Enhanced preprocessing for better edge detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return gray

def find_document_contour(edged):
    """Find the document contour using multiple methods"""
    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    
    # Filter out invalid contours and sort by area
    valid_cnts = []
    for c in cnts:
        try:
            area = cv2.contourArea(c)
            if area > 1000:  # Filter out very small contours
                valid_cnts.append(c)
        except:
            continue
    
    if not valid_cnts:
        print("No valid contours found!")
        return None
    
    # Sort by area and take top 5
    valid_cnts = sorted(valid_cnts, key=cv2.contourArea, reverse=True)[:5]
    
    # Method 1: Look for 4-sided contours
    for c in valid_cnts:
        try:
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                return approx
        except:
            continue
    
    # Method 2: If no 4-sided contour found, try with different epsilon
    for c in valid_cnts:
        try:
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            for epsilon in [0.01, 0.015, 0.025, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(c, epsilon * peri, True)
                if len(approx) == 4:
                    return approx
        except:
            continue
    
    # Method 3: If still no 4-sided contour, try to create one from the largest contour
    if valid_cnts:
        largest_contour = valid_cnts[0]
        try:
            # Get bounding rectangle and use its corners
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            return box.reshape(-1, 1, 2)
        except:
            pass
    
    return None

def save_image(image, filename, folder="output"):
    """Save image to specified folder"""
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, image)
    print(f"Saved: {filepath}")

def process_single_image(image_path, show_steps=False):
    """Process a single image"""
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print('='*60)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        return False
    
    # Get filename without extension for output naming
    base_name = image_path.stem
    
    # Compute the ratio of the old height to the new height
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    
    # Enhanced preprocessing
    gray = enhance_image_preprocessing(image)
    
    # Edge detection with multiple thresholds for better results
    edged1 = cv2.Canny(gray, 50, 150)
    edged2 = cv2.Canny(gray, 75, 200)
    edged3 = cv2.Canny(gray, 100, 250)
    
    # Combine edge maps
    edged = cv2.bitwise_or(edged1, edged2)
    edged = cv2.bitwise_or(edged, edged3)
    
    # Morphological operations to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    print("✓ STEP 1: Edge Detection Complete")
    
    # Save edge detection result
    save_image(edged, f"{base_name}_01_edge_detection.jpg")
    
    if show_steps:
        cv2.imshow("Original", image)
        cv2.imshow("Edged", edged)
        cv2.waitKey(0)
    
    # Find document contour
    try:
        screenCnt = find_document_contour(edged)
    except Exception as e:
        print(f"Error finding contour: {e}")
        print("Trying alternative edge detection...")
        
        # Try simpler edge detection
        edged_simple = cv2.Canny(gray, 50, 200)
        save_image(edged_simple, f"{base_name}_01_edge_detection_simple.jpg")
        screenCnt = find_document_contour(edged_simple)
    
    if screenCnt is None:
        print("✗ Could not find document contour!")
        print("  Possible reasons:")
        print("  - Document edges are not clear enough")
        print("  - Background too similar to document")
        print("  - Document is not rectangular")
        print("  - Try with better lighting or different background")
        return False
    
    print("✓ STEP 2: Document Boundary Found")
    
    # Draw the contour on the image
    boundary_image = image.copy()
    cv2.drawContours(boundary_image, [screenCnt], -1, (0, 255, 0), 2)
    save_image(boundary_image, f"{base_name}_02_boundary_detection.jpg")
    
    if show_steps:
        cv2.imshow("Boundary", boundary_image)
        cv2.waitKey(0)
    
    # Apply perspective transform
    try:
        if len(screenCnt) == 4:
            # Scale the contour back to original image size
            screenCnt_scaled = screenCnt.reshape(4, 2) * ratio
            
            # Apply four point transform
            warped = four_point_transform(orig, screenCnt_scaled)
            
            # Convert to grayscale and apply threshold for clean document
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            warped_thresh = cv2.adaptiveThreshold(warped_gray, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 10)
            
            print("✓ STEP 3: Perspective Transform Applied")
            
            # Save results
            save_image(warped, f"{base_name}_03_scanned_color.jpg")
            save_image(warped_thresh, f"{base_name}_04_scanned_bw.jpg")
            
            if show_steps:
                cv2.imshow("Original", orig)
                cv2.imshow("Scanned Color", imutils.resize(warped, height=650))
                cv2.imshow("Scanned B&W", imutils.resize(warped_thresh, height=650))
                cv2.waitKey(0)
        
        else:
            print(f"⚠ Warning: Found {len(screenCnt)} points instead of 4. Saving boundary detection only.")
            print("  The detected contour might not be a perfect rectangle.")
    
    except Exception as e:
        print(f"✗ Error in perspective transform: {e}")
        print("  Saving boundary detection only.")
        return False
    
    print("✓ Processing completed successfully!")
    return True

def main():
    # Create directories
    create_directories()
    
    # Check if images folder has any files
    images_folder = Path("images")
    image_files = list(images_folder.glob("*.jpg")) + list(images_folder.glob("*.png")) + list(images_folder.glob("*.jpeg"))
    
    if not image_files:
        print("\n" + "="*60)
        print("NO IMAGES FOUND!")
        print("="*60)
        print("Please add image files to the 'images' folder first.")
        print("Supported formats: .jpg, .jpeg, .png")
        print("\nUsage:")
        print("  python scan.py                    # Process all images")
        print("  python scan.py -i images/file.jpg # Process single image")
        print("  python scan.py --show             # Process all with preview")
        print("="*60)
        return
    
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Enhanced Document Scanner")
    ap.add_argument("-i", "--image", required=False,
                    help="Path to specific image to be scanned")
    ap.add_argument("--show", action="store_true",
                    help="Show intermediate processing steps")
    args = vars(ap.parse_args())
    
    # Determine what to process
    if args["image"]:
        # Process single image
        image_path = Path(args["image"])
        if not image_path.exists():
            print(f"Error: Image file '{args['image']}' not found!")
            return
        
        success = process_single_image(image_path, args["show"])
        if success:
            print(f"\n✓ Successfully processed: {image_path.name}")
        else:
            print(f"\n✗ Failed to process: {image_path.name}")
    
    else:
        # Process all images
        print(f"\n🚀 AUTO-PROCESSING MODE")
        print(f"Found {len(image_files)} image(s) in images folder:")
        for img in image_files:
            print(f"  📄 {img.name}")
        print()
        
        successful = 0
        failed = 0
        
        for image_file in image_files:
            success = process_single_image(image_file, args["show"])
            if success:
                successful += 1
            else:
                failed += 1
            
            if args["show"]:
                cv2.destroyAllWindows()
        
        # Summary
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Successfully processed: {successful} images")
        print(f"✗ Failed to process: {failed} images")
        print(f"📁 Total processed: {successful + failed} images")
        print(f"📂 Check 'output' folder for results!")
        print(f"{'='*60}")
    
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()