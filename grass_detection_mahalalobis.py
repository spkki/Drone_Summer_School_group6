import numpy as np
import cv2
from scipy.spatial.distance import mahalanobis
from scipy import linalg

##FOR PICTURE
image_path = 'Codebase/assets/capture_4/img_10.jpg'
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not read image")
    exit()
height, width = frame.shape[:2]

def sample_grass_from_image(image_path):
    """Automatically sample grass colors from the entire reference image
    Args:
        image_path: Path to reference image with good grass examples
    Returns:
        Array of HSV grass samples
    """
    ref_img = cv2.imread(image_path)
    if ref_img is None:
        print(f"Could not load reference image: {image_path}")
        return None  # Fall back will be handled by caller
    
    print(f"Automatically sampling grass from: {image_path}")
    ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    
    # Sample from the entire image since it's all grass
    # Take every 5th pixel to get good variety without too many samples
    grass_samples = ref_hsv[::5, ::5].reshape(-1, 3)
    
    print(f"Collected {len(grass_samples)} grass samples from reference image")
    return grass_samples

def create_custom_grass_samples():
    """Create grass samples from your reference grass image"""
    
    # Use the Grassimage.JPG as reference since it has no animals
    grass_ref_path = 'Codebase/assets/capture_4/img_3.jpg'
    
    grass_samples = sample_grass_from_image(grass_ref_path)
    if grass_samples is not None:
        print(f"Successfully loaded grass reference from {grass_ref_path}")
        return grass_samples
    else:
        print("Could not load grass reference, using default samples")
        return None

def create_grass_samples():
    """Create representative grass samples for Mahalanobis distance calculation"""
    # Try to use custom samples from your reference image
    grass_samples = create_custom_grass_samples()
    if grass_samples is not None:
        return grass_samples
        
    # Fallback to default samples if reference image fails
    print("Using fallback default grass samples")
    grass_samples_hsv = np.array([
        # More diverse green grass variations
        [25, 40, 60], [30, 50, 80], [35, 60, 100], [40, 70, 120], [45, 80, 140],
        [50, 90, 160], [55, 100, 180], [60, 110, 200],
        # Brown/tan grass and soil variations  
        [10, 20, 40], [15, 30, 60], [20, 40, 80], [25, 50, 100],
        [5, 15, 30], [8, 25, 50], [12, 35, 70], [18, 45, 90],
        # Dead/dry grass
        [30, 30, 70], [35, 35, 90], [40, 40, 110], [45, 45, 130],
        # Very light variations (sandy soil, dry areas)
        [20, 10, 100], [25, 15, 120], [30, 20, 140], [35, 25, 160],
    ])
    return grass_samples_hsv

def mahalanobis_grass_detection(hsv_img, threshold=2.5):
    """Detect grass using Mahalanobis distance"""
    grass_samples = create_grass_samples()
    mean_grass = np.mean(grass_samples, axis=0)
    cov_grass = np.cov(grass_samples.T)
    
    # Add regularization to handle singular matrices
    regularization = 1e-4
    cov_grass += np.eye(cov_grass.shape[0]) * regularization
    
    try:
        inv_cov = linalg.inv(cov_grass)
    except linalg.LinAlgError:
        inv_cov = linalg.pinv(cov_grass)
    
    # Reshape and calculate distances
    h, w, _ = hsv_img.shape
    pixels = hsv_img.reshape(-1, 3).astype(np.float32)
    diff = pixels - mean_grass
    distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    
    # Create mask
    mask = (distances < threshold).reshape(h, w).astype(np.uint8) * 255
    return mask

def adaptive_grass_detection(hsv_img):
    """Try multiple approaches to find the best grass detection for this image"""
    
    # Approach 1: Mahalanobis with different thresholds
    thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    best_mask = None
    best_coverage = 0
    
    for thresh in thresholds:
        mask = mahalanobis_grass_detection(hsv_img, threshold=thresh)
        coverage = (np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])) * 100
        
        print(f"Threshold {thresh}: {coverage:.1f}% grass coverage")
        
        # Look for reasonable grass coverage (20-80%)
        if 20 <= coverage <= 80:
            best_mask = mask
            best_coverage = coverage
            print(f"Using threshold {thresh} with {coverage:.1f}% coverage")
            break
        elif coverage > best_coverage and coverage < 90:
            best_mask = mask
            best_coverage = coverage
    
    # Approach 2: If Mahalanobis fails, fall back to HSV ranges
    if best_mask is None or best_coverage < 5:
        print("Mahalanobis failed, falling back to HSV range detection...")
        
        # Try multiple HSV ranges
        hsv_ranges = [
            ([20, 20, 20], [80, 255, 255]),    # Broad green range
            ([0, 0, 20], [180, 100, 200]),     # Very broad soil range
            ([10, 10, 10], [100, 255, 220]),   # Mixed range
        ]
        
        for i, (lower, upper) in enumerate(hsv_ranges):
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
            coverage = (np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])) * 100
            
            print(f"HSV range {i+1}: {coverage:.1f}% grass coverage")
            
            if 20 <= coverage <= 80:
                best_mask = mask
                best_coverage = coverage
                print(f"Using HSV range {i+1} with {coverage:.1f}% coverage")
                break
            elif coverage > best_coverage and coverage < 90:
                best_mask = mask
                best_coverage = coverage
    
    return best_mask if best_mask is not None else np.zeros(hsv_img.shape[:2], dtype=np.uint8)

# Process the frame using adaptive grass detection
print("Applying adaptive grass detection...")
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = adaptive_grass_detection(hsv)

# Apply more moderate morphological operations to preserve animal regions
# Use smaller kernels to avoid over-connecting grass areas
kernel_small = np.ones((3,3), np.uint8)
kernel_medium = np.ones((7,7), np.uint8)  # Reduced from 9x9
kernel_large = np.ones((15,15), np.uint8)  # Reduced from 21x21

# First, close small gaps
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)

# Remove very small noise
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

# Moderately connect grass regions (less aggressive)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

# Skip the massive closing that was connecting everything
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_xlarge)  # REMOVED

# Light dilation only
mask = cv2.dilate(mask, kernel_small, iterations=1)  # Reduced from medium kernel, 2 iterations

# More moderate opening to preserve animal-sized gaps
kernel_moderate = np.ones((7,7), np.uint8)  # Reduced from 15x15
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_moderate)

#Invert the mask to get the non-green areas
mask_inv = cv2.bitwise_not(mask)  #Invert the mask to get non-green areas

# More moderate filtering of the inverted mask to preserve animal-sized regions
# Remove small holes/objects but preserve medium-sized animal regions
kernel_cleanup = np.ones((5,5), np.uint8)  # Reduced from 11x11
kernel_final = np.ones((3,3), np.uint8)    # Reduced from 7x7

# Remove small isolated non-grass areas (less aggressive opening)
mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel_cleanup)

# Close small gaps within animal regions
mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel_final)

# Skip erosion to preserve animal boundaries
# mask_inv = cv2.erode(mask_inv, kernel_small, iterations=1)  # REMOVED

# Debug: count non-grass regions
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_inv)
print(f"Found {num_labels-1} non-grass regions before shape analysis")

def analyze_shape_features(contour):
    """Analyze shape features to determine if it's likely an animal"""
    area = cv2.contourArea(contour)
    if area < 200:  # Reduced minimum size to catch smaller animals
        return False, 0
    
    # Calculate multiple shape descriptors
    perimeter = cv2.arcLength(contour, True)
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    
    # Shape metrics
    aspect_ratio = w / h
    extent = area / rect_area if rect_area > 0 else 0  # How much of bounding box is filled
    
    # Handle small contours that might not have proper hulls
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0  # Convexity
    
    # Circularity (4*pi*area / perimeter^2) - closer to 1 is more circular
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0
    
    # More permissive animal shape criteria for smaller animals
    is_animal = (
        0.2 <= aspect_ratio <= 5.0 and      # More lenient aspect ratio
        extent > 0.2 and                     # Lower extent threshold
        solidity > 0.3 and                   # More lenient solidity
        0.05 <= circularity <= 0.95 and     # Wider circularity range
        area >= 300                          # Lower minimum size
    )
    
    # Calculate confidence score (adjusted for new criteria)
    confidence = (
        min(aspect_ratio, 1/aspect_ratio) * 0.3 +  # Less penalty for ratios
        extent * 0.4 +                             # Moderate reward for fill
        solidity * 0.4 +                           # Moderate reward for convexity
        (1 - abs(circularity - 0.4)) * 0.3         # Reward for reasonable circularity
    )
    
    # Boost confidence for medium-sized objects (likely animals)
    if 500 <= area <= 50000:
        confidence *= 1.2
    
    return is_animal, min(confidence, 1.0)  # Cap confidence at 1.0
    
result = cv2.bitwise_and(frame, frame, mask=mask_inv) #Change mask depending on what you want to see

#Draw circles around detected items with improved animal detection
contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} potential objects")
animal_detections = []

for i, contour in enumerate(contours):
    is_animal, confidence = analyze_shape_features(contour)
    
    if is_animal:
        # Get bounding info
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        area = cv2.contourArea(contour)
        
        # Color based on confidence: green for high confidence, yellow for medium
        if confidence > 0.7:
            color = (0, 255, 0)  # Green - high confidence
            thickness = 3
        elif confidence > 0.5:
            color = (0, 255, 255)  # Yellow - medium confidence  
            thickness = 2
        else:
            color = (0, 165, 255)  # Orange - low confidence
            thickness = 1
            
        cv2.circle(result, center, radius, color, thickness)
        
        # Add detailed info
        cv2.putText(result, f'Animal {i+1}', (center[0]-30, center[1]-radius-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(result, f'Conf: {confidence:.2f}', (center[0]-30, center[1]-radius-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(result, f'Area: {int(area)}', (center[0]-30, center[1]+radius+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        animal_detections.append({
            'center': center,
            'radius': radius,
            'area': area,
            'confidence': confidence
        })

print(f"Detected {len(animal_detections)} potential animals")
for i, detection in enumerate(animal_detections):
    print(f"Animal {i+1}: Confidence={detection['confidence']:.2f}, Area={detection['area']}")
            
#Create an empty image to display results
image = np.zeros(frame.shape, np.uint8)  # Create an empty image to display results
smaller_original = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize the original frame to half its size
smaller_hsv = cv2.resize(hsv, (0, 0), fx=0.5, fy=0.5)  # Resize the HSV frame to half its size
smaller_mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)  # Resize the mask to half its size
smaller_mask_3ch = cv2.cvtColor(smaller_mask, cv2.COLOR_GRAY2BGR)  # Convert mask to 3-channel for display
smaller_result = cv2.resize(result, (0, 0), fx=0.5, fy=0.5)  # Resize the result frame to half its size
    
#Combine the resized images into one
image[:height//2, :width//2] = smaller_original  # Place the original frame in the top-left corner
image[height//2:, :width//2] = smaller_mask_3ch  # Place the mask in the bottom-left corner
image[:height//2, width//2:] = smaller_hsv  # Place the HSV frame in the top-right corner
image[height//2:, width//2:] = smaller_result  # Place the result frame in the bottom-right corner
    
#Text annotations for each section
image = cv2.putText(image, 'Mask', (10, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Add text to the image
image = cv2.putText(image, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Add text to the image
image = cv2.putText(image, 'HSV', (width // 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Add text to the image
image = cv2.putText(image, 'Result', (width // 2 + 10, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Add text to the image

image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)  # Resize the combined image to half its size
cv2.imshow("Combined", image)  # Show the combined image

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()