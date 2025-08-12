import numpy as np
import cv2

##FOR PICTURE
image_path = 'Codebase/assets/capture_4/img_10.jpg'
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

# Very aggressive filtering of the inverted mask to remove small non-grass regions
# Remove small holes/objects that are too small to be animals
kernel_cleanup = np.ones((11,11), np.uint8)
kernel_final = np.ones((7,7), np.uint8)

# Remove small isolated non-grass areas (aggressive opening)
mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel_cleanup)

# Close small gaps within animal regions
mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel_final)

# Final erosion to clean up edges
mask_inv = cv2.erode(mask_inv, kernel_small, iterations=1)
    
result = cv2.bitwise_and(frame, frame, mask=mask_inv) #Change mask depending on what you want to see

#Draw circles around detected items in non-grass areas (potential animals)
contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    # Increased threshold for animal detection - filters out small objects
    if area > 2000:  # Much higher threshold for animal-sized objects
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Additional filter: check aspect ratio to avoid very thin/long objects
        rect = cv2.boundingRect(contour)
        aspect_ratio = rect[2] / rect[3]  # width/height
        
        # Only draw circle if aspect ratio is reasonable for animals (not too elongated)
        if 0.3 <= aspect_ratio <= 3.0:  # Reasonable range for animal shapes
            cv2.circle(result, center, radius, (0, 255, 0), 2)
            # Add area text for debugging
            cv2.putText(result, f'Area: {int(area)}', (center[0]-30, center[1]-radius-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
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