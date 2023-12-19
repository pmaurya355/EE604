import cv2
import numpy as np

def solution(image_path):
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    gray_inverted = cv2.bitwise_not(gray)
    
    # Apply thresholding
    _, threshold = cv2.threshold(gray_inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find coordinates of non-zero pixels
    coords = np.column_stack(np.where(threshold != 0))
    
    # Get the angle of the minimum bounding rectangle
    angle = cv2.minAreaRect(coords)[-1]
    
    # Correct the angle if it's close to -45 degrees
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Ensure the angle is within a reasonable range
    if angle < -45:
        angle += 90
    if abs(angle) > 135:
        angle += 180

    # Get the center of the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Apply the rotation to the image
    matrix = cv2.getRotationMatrix2D(center, angle, 0.75)
    image_output = cv2.warpAffine(image, matrix, (w , h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue = (255, 255, 255))
    
    # cv2.imshow("Transformed Image", image_output)
    # cv2.waitKey(0)
        
    return image_output
