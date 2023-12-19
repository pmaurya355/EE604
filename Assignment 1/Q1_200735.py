import cv2  
import numpy as np  

def get_image_corners(image):
    # Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to the grayscale image
    threshold = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

    # Find external contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # Calculate epsilon
        e = 0.05 * cv2.arcLength(c, True)

        # Approximate the contour with a polygon
        approx = cv2.approxPolyDP(c, e, True)

        # Quadrilateral found
        if len(approx) == 4:
            return approx  
    return None  

def solution(image_path):
    image = cv2.imread(image_path)

    image_corners = get_image_corners(image)

    # Sort the corners in clockwise manner
    image_corners = np.array(sorted(np.concatenate(image_corners).tolist()))

    sorted_corners = np.zeros((4, 2), dtype='float32')

    # Calculate differences and sums of x and y coordinates to sort corners
    d = np.diff(image_corners, axis=1)
    s = image_corners.sum(axis=1)
    sorted_corners[0] = image_corners[np.argmin(s)]
    sorted_corners[1] = image_corners[np.argmin(d)]
    sorted_corners[2] = image_corners[np.argmax(s)]
    sorted_corners[3] = image_corners[np.argmax(d)]

    # Output quadrilateral with specific coordinates
    output = np.float32([[0, 0], [599, 0], [599, 599], [0, 599]])

    # Get the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(sorted_corners, output)

    # Apply the perspective transformation to the image
    transformed_image = cv2.warpPerspective(image, matrix, (600, 600))

    # cv2.imshow("Transformed Image", transformed_image)
    # cv2.waitKey(0)

    return transformed_image 
