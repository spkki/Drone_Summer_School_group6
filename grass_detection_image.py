import numpy as np
import cv2

##FOR PICTURE
image_path = 'assets/capture_4/img_10.jpg'
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not read image")
    exit()
height, width = frame.shape[:2]

# Process the frame (detect grass)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Use a single, broader range for natural grass/ground detection
# This covers green grass, brown soil, and dead vegetation
lower_ground = np.array([0, 0, 10])     # Very broad range for natural ground
upper_ground = np.array([95, 255, 200]) # Covers green to brown hues

# Create initial mask
mask = cv2.inRange(hsv, lower_ground, upper_ground)

# Apply aggressive morphological operations to clean up and connect regions
# Use progressively larger kernels for better grass connectivity
kernel_small = np.ones((3,3), np.uint8)
kernel_medium = np.ones((9,9), np.uint8)
kernel_large = np.ones((21,21), np.uint8)
kernel_xlarge = np.ones((31,31), np.uint8)

# First, close small gaps
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)

# Remove very small noise
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

# Aggressively connect grass regions with larger kernels
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

# Final massive closing to create large connected grass areas
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_xlarge)

# Dilate to ensure good grass coverage
mask = cv2.dilate(mask, kernel_medium, iterations=2)

# Remove any remaining small grass patches that shouldn't be grass
# This aggressive opening will eliminate small isolated grass areas
kernel_aggressive = np.ones((15,15), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_aggressive)

#Invert the mask to get the non-green areas
mask_inv = cv2.bitwise_not(mask)  #Invert the mask to get non-green areas

# Add edge detection to help identify animal boundaries
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise before edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Dilate edges to make them more prominent
edge_kernel = np.ones((3,3), np.uint8)
edges = cv2.dilate(edges, edge_kernel, iterations=1)

# Combine edge information with non-grass areas
# Only keep edges that are in non-grass areas (potential animals)
edges_in_non_grass = cv2.bitwise_and(edges, mask_inv)

# Create a hybrid approach: start with non-grass areas and enhance with edges
# Apply light morphological operations to clean up the non-grass mask
kernel_cleanup = np.ones((5,5), np.uint8)  # Smaller kernel to preserve animals
kernel_final = np.ones((3,3), np.uint8)    # Smaller kernel

# Light cleanup of the inverted mask (non-grass areas)
cleaned_non_grass = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel_cleanup)
cleaned_non_grass = cv2.morphologyEx(cleaned_non_grass, cv2.MORPH_CLOSE, kernel_final)

# Remove only very small connected components from non-grass areas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_non_grass)
filtered_non_grass = np.zeros_like(cleaned_non_grass)
min_non_grass_area = 500  # Lower threshold to preserve more animals

for i in range(1, num_labels):  # Skip background (label 0)
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= min_non_grass_area:
        filtered_non_grass[labels == i] = 255

# Simple approach: use the filtered non-grass areas directly
# This is less aggressive and should preserve animals better
mask_inv = filtered_non_grass
    
result = cv2.bitwise_and(frame, frame, mask=mask_inv) #Change mask depending on what you want to see

#Draw circles around detected items in non-grass areas (potential animals)
contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} potential objects")
animal_count = 0

for contour in contours:
    area = cv2.contourArea(contour)
    # Lower threshold to catch more animals
    if area > 600:  # Reduced threshold
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Additional filters for animal detection
        rect = cv2.boundingRect(contour)
        aspect_ratio = rect[2] / rect[3]  # width/height
        
        # Calculate additional shape features
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0  # How convex the shape is
        
        # Calculate extent (area vs bounding rectangle area)
        rect_area = rect[2] * rect[3]
        extent = area / rect_area if rect_area > 0 else 0
        
        # More permissive filtering to preserve animals
        if (0.1 <= aspect_ratio <= 8.0 and  # Very permissive aspect ratio
            solidity > 0.2 and               # Very permissive solidity
            extent > 0.15 and                # Permissive extent
            area <= 40000 and                # Not too large
            area >= 600):                    # Lower minimum
            
            animal_count += 1
            
            # Color coding based on confidence
            if area > 2000 and 0.3 <= aspect_ratio <= 4.0 and solidity > 0.4:
                color = (0, 255, 0)  # Green for high confidence
                thickness = 3
            elif area > 1000 and solidity > 0.3:
                color = (0, 255, 255)  # Yellow for medium confidence
                thickness = 2
            else:
                color = (0, 165, 255)  # Orange for low confidence
                thickness = 1
            
            cv2.circle(result, center, radius, color, thickness)
            
            # Add detailed information
            cv2.putText(result, f'Animal {animal_count}', (center[0]-30, center[1]-radius-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(result, f'Area: {int(area)}', (center[0]-30, center[1]-radius-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(result, f'AR: {aspect_ratio:.2f}', (center[0]-30, center[1]+radius+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

print(f"Detected {animal_count} potential animals")
            
#Create an empty image to display results
image = np.zeros(frame.shape, np.uint8)  # Create an empty image to display results
smaller_original = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize the original frame to half its size
smaller_hsv = cv2.resize(hsv, (0, 0), fx=0.5, fy=0.5)  # Resize the HSV frame to half its size
smaller_mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)  # Resize the mask to half its size
smaller_mask_3ch = cv2.cvtColor(smaller_mask, cv2.COLOR_GRAY2BGR)  # Convert mask to 3-channel for display
smaller_result = cv2.resize(result, (0, 0), fx=0.5, fy=0.5)  # Resize the result frame to half its size

# Also show edges for debugging
smaller_edges = cv2.resize(edges_in_non_grass, (0, 0), fx=0.5, fy=0.5)  # Show edges in non-grass areas
smaller_edges_3ch = cv2.cvtColor(smaller_edges, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
    
#Combine the resized images into one (2x3 grid layout)
image[:height//2, :width//2] = smaller_original  # Place the original frame in the top-left corner
image[height//2:, :width//2] = smaller_mask_3ch  # Place the mask in the bottom-left corner
image[:height//2, width//2:] = smaller_hsv  # Place the HSV frame in the top-right corner
image[height//2:, width//2:] = smaller_result  # Place the result frame in the bottom-right corner

# Create a separate window for edges
smaller_edges_3ch = cv2.resize(smaller_edges_3ch, (0, 0), fx=0.5, fy=0.5)  # Resize edges for display
cv2.imshow("Edges", smaller_edges_3ch)
    
#Text annotations for each section
image = cv2.putText(image, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Add text to the image
image = cv2.putText(image, 'Grass Mask', (10, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Add text to the image
image = cv2.putText(image, 'HSV', (width // 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Add text to the image
image = cv2.putText(image, 'Result', (width // 2 + 10, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Add text to the image

image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)  # Resize the combined image to quarter size
cv2.imshow("Combined", image)  # Show the combined image

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()