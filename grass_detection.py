import numpy as np
import cv2

##FOR CAMERA
cap = cv2.VideoCapture(0)  # Open the default camera

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    width = int(cap.get(3))  # Get the width of the frame
    height = int(cap.get(4))  # Get the height of the frame
    if not ret:
        break

    # Process the frame (detect grass)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40]) #Green color lower
    upper_green = np.array([85, 255, 255])#Green color upper

    mask = cv2.inRange(hsv, lower_green, upper_green) #Mask for the green color
    #Invert the mask to get the non-green areas
    mask_inv = cv2.bitwise_not(mask)  #Invert the mask to get non-green areas
    
    
    result = cv2.bitwise_and(frame, frame, mask=mask_inv) #Change mask depending on what you want to see

    #Draw circles around detected items in non-grass areas (items on grass)
    contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust this threshold based on object size
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(result, center, radius, (0, 255, 0), 2)
            
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

    cv2.imshow("Combined", image)  # Show the combined image
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()