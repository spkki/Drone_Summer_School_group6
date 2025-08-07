import cv2

def findContoursInFile(filename):
    image = cv2.imread(filename, 1)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(image_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def analyzeContours(contours, objectClass):
    for idx, contour in enumerate(contours):
        perimeter = cv2.arcLength(contours[idx], 1)
        area = cv2.contourArea(contours[idx], True)
        if area <= 0:
            continue
        compactness = perimeter * perimeter / (4 * 3.141592 * area)

        mu = cv2.moments(contours[idx], False)
        hu = cv2.HuMoments(mu).flatten()
        print(f"{objectClass:d}\t", end='')
        print(f"{perimeter:.3f}\t{area:.0f}\t{compactness:.3f}", end='')

        print(f"\t{mu['mu20']:.0f}", end='')
        print(f"\t{mu['mu11']:.0f}", end='')
        print(f"\t{mu['mu02']:.0f}", end='')
        print(f"\t{mu['mu30']:.0f}", end='')
        print(f"\t{mu['mu21']:.0f}", end='')
        print(f"\t{mu['mu12']:.0f}", end='')
        print(f"\t{mu['mu03']:.0f}", end='')

        print(f"\t{hu[0]:.6f}", end='')
        print(f"\t{hu[1]:.6f}", end='')
        print(f"\t{hu[2]:.6f}", end='')
        print(f"\t{hu[3]:.10f}", end='')
        print(f"\t{hu[4]:.10f}", end='')
        print(f"\t{hu[5]:.12f}", end='')
        print(f"\t{hu[6]:.12f}")


def main():
    print("class\tperimeter\tarea\tcompactness\tmu20\tmu11\tmu02\tmu30\tmu21\tmu12\tmu03\thu1\thu2\thu3\thu4\thu5\thu6\thu7")
    analyzeContours(findContoursInFile("../input/numbers/1.png"), 1)
    analyzeContours(findContoursInFile("../input/numbers/2.png"), 2)
    analyzeContours(findContoursInFile("../input/numbers/3.png"), 3)
    analyzeContours(findContoursInFile("../input/numbers/4.png"), 4)
    analyzeContours(findContoursInFile("../input/numbers/5.png"), 5)
    analyzeContours(findContoursInFile("../input/numbers/6.png"), 6)
    analyzeContours(findContoursInFile("../input/numbers/7.png"), 7)
    analyzeContours(findContoursInFile("../input/numbers/8.png"), 8)
    analyzeContours(findContoursInFile("../input/numbers/9.png"), 9)


main()
