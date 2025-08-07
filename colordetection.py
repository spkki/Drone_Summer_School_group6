import numpy as np
import cv2

cap = cv2.VideoCapture(0)  # Open the default camera

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))  # Get the width of the frame
    height = int(cap.get(4))  # Get the height of the frame
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert the frame to HSV color space
    lower_blue = np.array([110, 50, 50])  # Define lower bound for blue color
    upper_blue = np.array([130, 255, 255])  # Define upper bound for blue color
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)  # Create a mask for blue color
    
    result = cv2.bitwise_and(frame, frame, mask=mask)  # Apply the mask to the original frame
    
    image = np.zeros(frame.shape, np.uint8)  # Create an empty image to display results
    smaller_original = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize the original frame to half its size
    smaller_hsv = cv2.resize(hsv, (0, 0), fx=0.5, fy=0.5)  # Resize the HSV frame to half its size
    smaller_mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)  # Resize the mask to half its size
    smaller_mask_3ch = cv2.cvtColor(smaller_mask, cv2.COLOR_GRAY2BGR)  # Convert mask to 3-channel for display
    smaller_result = cv2.resize(result, (0, 0), fx=0.5, fy=0.5)  # Resize the result frame to half its size
    image[:height//2, :width//2] = smaller_original  # Place the original frame in the top-left corner
    image[height//2:, :width//2] = smaller_hsv  # Place the HSV frame in the bottom-left corner
    image[:height//2, width//2:] = smaller_mask_3ch  # Place the mask in the top-right corner
    image[height//2:, width//2:] = smaller_result  # Place the result frame in the bottom-right corner
    image = cv2.putText(image, 'HSV', (10, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Add text to the image
    image = cv2.putText(image, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Add text to the image
    image = cv2.putText(image, 'Mask', (width // 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Add text to the image
    image = cv2.putText(image, 'Result', (width // 2 + 10, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Add text to the image

    #cv2.imshow('Original Frame', frame)  # Display the original frame
    #cv2.imshow('Mask', mask)  # Display the mask
    #cv2.imshow('Result', result)  # Display the result of the color detection
    #cv2.imshow('Camera Feed', hsv)  # Display the original frame
    
    cv2.imshow('Camera Feed', image)  # Display the processed image
    
    
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
