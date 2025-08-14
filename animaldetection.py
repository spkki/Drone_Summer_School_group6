# Import necessary libraries
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import cv2

# Setup for Mahalanobis distance calculation
# load image and convert to HSV
image = cv2.imread("assets/capture_3/img_5.jpg")
resized = cv2.resize(image, (800, 600))
cv2.imshow("Original Image", resized)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#cv2.imshow("HSV Image", hsv_image)

# Define the lower and upper limits for the green color in HSV and calculate the mask
lower_limit = np.array([0, 40, 40])
upper_limit = np.array([85, 255, 255])
mask = cv2.inRange(hsv_image, lower_limit, upper_limit) 

lower_yellow = np.array([15, 30, 50])
upper_yellow = np.array([35, 255, 255])
mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
mask_combined = cv2.bitwise_or(mask, mask_yellow)
#mean, std = cv2.meanStdDev(image, mask = mask)

# Work with pixels
pixels = np.reshape(image, (-1, 3))
mask_pixels = np.reshape(mask_combined, (-1))
annot_pix_values = pixels[mask_pixels == 255, ]
avg = np.average(annot_pix_values, axis=0)
cov = np.cov(annot_pix_values.transpose())

# Calculate the difference
shape = pixels.shape
avg_value = np.repeat([avg], shape[0], axis=0)
diff = pixels - avg_value

# Calculate Mahalanobis distance between for the green values
inv_cov = np.linalg.inv(cov)
moddotproduct = diff * (diff @ inv_cov)
mahalanobis_distance = np.sum(moddotproduct, axis=1)
mahalanobis_distance_image = np.reshape(mahalanobis_distance, (image.shape[0], image.shape[1]))

# Threshold: classify grass (distance < 3 = grass)
threshold = 85.0
grass_mask = (mahalanobis_distance_image < threshold).astype(np.uint8) * 255

# Remove grass
result = image.copy()
result[grass_mask == 255] = [0, 0, 0]  # black where grass was

resized_mahalanobis_distance_image = cv2.resize(mahalanobis_distance_image, (800, 600))
resized_grass_mask = cv2.resize(grass_mask, (800, 600))
resized_result = cv2.resize(result, (800, 600))

# Show results
#cv2.imshow("Mahalanobis Distance", resized_mahalanobis_distance_image / resized_mahalanobis_distance_image.max())
#cv2.imshow("Grass Mask", resized_grass_mask)
cv2.imshow("Grass Removed", resized_result)

# Invert mask to detect non-grass objects
non_grass_mask = cv2.bitwise_not(grass_mask)

# Find contours in the non-grass mask
contours, _ = cv2.findContours(non_grass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
color_threshold = 15  # smaller = stricter match

out = image.copy()

Pixel_scale = 1250 # User defined
Altitude = 25 # meters
min_diameter = Pixel_scale / Altitude  # pixels

for i, cnt in enumerate(contours, start=1):
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    diameter = radius * 2
    if diameter < min_diameter:
        continue  # Skip small objects

    # Mask for this contour
    single_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(single_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Get HSV pixels inside the contour
    pixels_in_mask = hsv_image[single_mask == 255]
    if len(pixels_in_mask) == 0:
        continue

    # Average color in HSV
    avg_hsv = np.mean(pixels_in_mask, axis=0)
    avg_hsv_int = tuple(int(v) for v in avg_hsv)

    # Print the average color to terminal
    print(f"Contour {i}: Avg HSV = {avg_hsv_int}")

    # Draw a circle and label the contour number
    center = (int(x), int(y))
    #cv2.circle(out, center, int(radius), (0, 255, 0), 2)
    #cv2.putText(out, f"#{i}", center,
    #            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
    
    #Elephant
    color_E_up = np.array([116, 200, 95])
    color_E_low = np.array([112, 161, 56])

    if (avg_hsv[0] >= color_E_low[0] and avg_hsv[0] <= color_E_up[0] and
        avg_hsv[1] >= color_E_low[1] and avg_hsv[1] <= color_E_up[1] and
        avg_hsv[2] >= color_E_low[2] and avg_hsv[2] <= color_E_up[2]):
        cv2.circle(out, center, int(radius), (0, 255, 0), 2)
        cv2.putText(out, "Elephant", center, cv2.FONT_HERSHEY_SIMPLEX, 2, (116, 172, 191), 3)
    
    #Antelope    
    color_A_up = np.array([116, 177, 192])
    color_A_low = np.array([107, 144, 151])
    
    # Check if color matches antelope range
    antelope_color_match = (avg_hsv[0] >= color_A_low[0] and avg_hsv[0] <= color_A_up[0] and
                           avg_hsv[1] >= color_A_low[1] and avg_hsv[1] <= color_A_up[1] and
                           avg_hsv[2] >= color_A_low[2] and avg_hsv[2] <= color_A_up[2])
    
    #Hippo    
    color_H_up = np.array([115, 213, 121])
    color_H_low = np.array([111, 200, 114])

    if (avg_hsv[0] >= color_H_low[0] and avg_hsv[0] <= color_H_up[0] and
        avg_hsv[1] >= color_H_low[1] and avg_hsv[1] <= color_H_up[1] and
        avg_hsv[2] >= color_H_low[2] and avg_hsv[2] <= color_H_up[2]):
        cv2.circle(out, center, int(radius), (0, 255, 0), 2)
        cv2.putText(out, "Hippo", center, cv2.FONT_HERSHEY_SIMPLEX, 2, (116, 172, 191), 3) 
    
    #Lion
    color_L_up = np.array([141, 95, 142])
    color_L_low = np.array([121, 47, 101])

    if (avg_hsv[0] >= color_L_low[0] and avg_hsv[0] <= color_L_up[0] and
        avg_hsv[1] >= color_L_low[1] and avg_hsv[1] <= color_L_up[1] and
        avg_hsv[2] >= color_L_low[2] and avg_hsv[2] <= color_L_up[2]):
        cv2.circle(out, center, int(radius), (0, 255, 0), 2)
        cv2.putText(out, "Lion", center, cv2.FONT_HERSHEY_SIMPLEX, 2, (116, 172, 191), 3)
        
    #Zebra
    color_Z_up = np.array([118, 187, 172])
    color_Z_low = np.array([109, 135, 143])
    
    # Check if color matches zebra range
    zebra_color_match = (avg_hsv[0] >= color_Z_low[0] and avg_hsv[0] <= color_Z_up[0] and
                        avg_hsv[1] >= color_Z_low[1] and avg_hsv[1] <= color_Z_up[1] and
                        avg_hsv[2] >= color_Z_low[2] and avg_hsv[2] <= color_Z_up[2])
    
    # Size-based discrimination between zebra and antelope (both have similar colors)
    # Define size thresholds - adjust these based on your altitude and expected animal sizes
    large_animal_threshold = min_diameter * 2.5  # Zebras are typically larger
    small_animal_threshold = min_diameter * 0.8  # Antelopes are typically smaller
    
    # If both zebra and antelope color ranges match, use size to decide
    if zebra_color_match and antelope_color_match:
        if diameter >= large_animal_threshold:
            cv2.circle(out, center, int(radius), (0, 255, 0), 2)
            cv2.putText(out, "Zebra", center, cv2.FONT_HERSHEY_SIMPLEX, 2, (116, 172, 191), 3)
        elif diameter >= small_animal_threshold:
            cv2.circle(out, center, int(radius), (0, 255, 0), 2)
            cv2.putText(out, "Antelope", center, cv2.FONT_HERSHEY_SIMPLEX, 2, (116, 172, 191), 3)
    # If only zebra color matches
    elif zebra_color_match:
        cv2.circle(out, center, int(radius), (0, 255, 0), 2)
        cv2.putText(out, "Zebra", center, cv2.FONT_HERSHEY_SIMPLEX, 2, (116, 172, 191), 3)
    # If only antelope color matches
    elif antelope_color_match:
        cv2.circle(out, center, int(radius), (0, 255, 0), 2)
        cv2.putText(out, "Antelope", center, cv2.FONT_HERSHEY_SIMPLEX, 2, (116, 172, 191), 3)            
        
# Show labeled clusters
resized_out = cv2.resize(out, (800, 600))
cv2.imshow("Animals marked", resized_out)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
