import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import io


class ImagePreprocessor:
    """
    Image preprocessing pipeline for crop leaf disease detection.
    Handles color correction, background removal, and leaf segmentation.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.lower_green = np.array([25, 40, 40])
        # HSV color range for green leaves 
        self.upper_green = np.array([90, 255, 255])  
        
        # Alternate ranges for different leaf colors
        self.lower_green_alt = np.array([35, 50, 50])
        self.upper_green_alt = np.array([85, 255, 255])
    
    
    def color_correction(self, image: np.ndarray) -> np.ndarray:
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split into channels
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L-channel (luminance)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel_clahe = clahe.apply(l_channel)
        
        # Merge channels back
        lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
        
        # Convert back to BGR
        enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return enhanced_image
    
    
    def remove_background(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Step 1: Convert BGR to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Step 2: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # Step 3: Create binary mask using color thresholding
        mask = cv2.inRange(blurred, self.lower_green, self.upper_green)
        
       
        mask_alt = cv2.inRange(blurred, self.lower_green_alt, self.upper_green_alt)
        mask = cv2.bitwise_or(mask, mask_alt)
        
        # Step 4: Apply morphological operations to clean up the mask
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        mask = cv2.erode(mask, kernel_small, iterations=1)
        
        mask = cv2.dilate(mask, kernel_small, iterations=1)
        
        # Step 5: Optional - Fill remaining holes using floodFill
        mask_floodfill = mask.copy()
        h, w = mask.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(mask_floodfill, flood_mask, (0, 0), 255)
        mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)
        mask = mask | mask_floodfill_inv
        
        # Apply mask to original image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        return masked_image, mask
    
    
    def segment_leaf(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        # Step 1: Find contours in the binary mask
        contours, hierarchy = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Check if any contours were found
        if not contours:
            return None
        
        # Step 2: Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        min_contour_area = 1000 
        if cv2.contourArea(largest_contour) < min_contour_area:
            return None
        
        # Step 3: Get bounding rectangle around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Step 4: Add padding around the bounding box
        padding = int(0.1 * max(w, h))
        
        # Calculate padded coordinates (ensure within image bounds)
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(image.shape[1] - x_padded, w + 2 * padding)
        h_padded = min(image.shape[0] - y_padded, h + 2 * padding)
        
        # Step 5: Crop the image to the bounding box
        cropped_leaf = image[y_padded:y_padded + h_padded, 
                            x_padded:x_padded + w_padded]
        
        # Validate cropped image
        if cropped_leaf.size == 0:
            return None
        
        return cropped_leaf
    
    
    def resize_and_normalize(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        
        return normalized
    
    
    def preprocess(self, image_bytes: bytes) -> Tuple[np.ndarray, dict]:
        # Step 0: Decode image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image. Ensure the file is a valid JPEG or PNG.")
        
        original_height, original_width = image.shape[:2]
        metadata = {
            "original_size": f"{original_width}x{original_height}",
            "steps_completed": []
        }
        
        # Step 1: Color Correction
        corrected_image = self.color_correction(image)
        metadata["steps_completed"].append("color_correction")
        
        # Step 2: Background Removal
        masked_image, binary_mask = self.remove_background(corrected_image)
        metadata["steps_completed"].append("background_removal")
        
        leaf_pixel_count = np.count_nonzero(binary_mask)
        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
        leaf_area_percentage = (leaf_pixel_count / total_pixels) * 100
        metadata["leaf_area_percentage"] = round(leaf_area_percentage, 2)
        
        # Step 3: Leaf Segmentation
        segmented_leaf = self.segment_leaf(masked_image, binary_mask)
        
        if segmented_leaf is None or segmented_leaf.size == 0:
            raise ValueError(
                "No leaf region detected in the image. "
                "Ensure the image contains a clearly visible leaf with good lighting."
            )
        
        metadata["steps_completed"].append("leaf_segmentation")
        
        # Step 4: Resize and Normalize
        processed_image = self.resize_and_normalize(segmented_leaf)
        metadata["steps_completed"].append("resize_normalize")
        metadata["processed_size"] = f"{self.target_size[0]}x{self.target_size[1]}"
        
        # Step 5: Add batch dimension for model input
        batch_image = np.expand_dims(processed_image, axis=0)
        
        return batch_image, metadata
    
    
    def preprocess_with_debug(self, image_bytes: bytes, save_path: str = None) -> Tuple[np.ndarray, dict]:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Step 1: Color correction
        corrected = self.color_correction(image)
        
        # Step 2: Background removal
        masked, mask = self.remove_background(corrected)
        
        # Step 3: Segmentation
        segmented = self.segment_leaf(masked, mask)
        
        # Step 4: Resize
        if segmented is not None:
            processed = self.resize_and_normalize(segmented)
        else:
            raise ValueError("No leaf detected")
        
        # Save intermediate results if path provided
        if save_path:
            cv2.imwrite(f"{save_path}_1_corrected.jpg", corrected)
            cv2.imwrite(f"{save_path}_2_masked.jpg", masked)
            cv2.imwrite(f"{save_path}_3_mask.jpg", mask)
            if segmented is not None:
                cv2.imwrite(f"{save_path}_4_segmented.jpg", segmented)
            
            # Save final processed
            final_vis = (processed * 255).astype(np.uint8)
            final_vis = cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{save_path}_5_final.jpg", final_vis)
        
        
        batch_image = np.expand_dims(processed, axis=0)
        metadata = {
            "original_size": f"{image.shape[1]}x{image.shape[0]}",
            "processed_size": f"{self.target_size[0]}x{self.target_size[1]}",
            "steps_completed": ["color_correction", "background_removal", 
                              "segmentation", "resize_normalize"]
        }
        
        return batch_image, metadata


# Utility function for batch processing
def batch_preprocess(image_paths: list, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    preprocessor = ImagePreprocessor(target_size=target_size)
    processed_images = []
    
    for img_path in image_paths:
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        
        try:
            processed, _ = preprocessor.preprocess(img_bytes)
            processed_images.append(processed[0])  # Remove batch dimension
        except ValueError as e:
            print(f"Warning: Skipping {img_path} - {str(e)}")
            continue
    
    
    if processed_images:
        return np.array(processed_images)
    else:
        raise ValueError("No valid images processed")
