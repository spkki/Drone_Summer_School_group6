import cv2
import random

def main():
    filename = '../input/shapes.png'
    image = cv2.imread(filename)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(image_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.namedWindow("Display Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Display Image", image_grey)

    drawing = 0 * image
    random.seed(12345)
    for idx, contour in enumerate(contours):
        perimeter = cv2.arcLength(contours[idx], 1)
        area = cv2.contourArea(contours[idx], True)
        if area < 0:
            continue
        compactness = perimeter * perimeter / (4 * 3.141592 * area)
        print(f'Perimeter:   {perimeter:8.3f}')
        print(f'Area:        {area:8.3f}')
        print(f'Compactness: {area:8.3f}')

        moments = cv2.moments(contours[idx], False)
        hu_moments = cv2.HuMoments(moments)
        print(hu_moments)
    
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.drawContours(drawing, contours, idx, color, 1, 8, hierarchy, 0)

    cv2.namedWindow("Contours", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Contours", drawing)
    cv2.waitKey(-1)

main()
