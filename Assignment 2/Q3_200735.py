import cv2
import numpy as np

def solution(audio_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    # class_name = 'fake'

    image = cv2.imread(audio_path, cv2.IMREAD_GRAYSCALE)

    # Identify the first and last non-white rows
    non_white_rows = np.where(image < 250)[0]
    first_non_white_row = non_white_rows[0]
    last_non_white_row = non_white_rows[-1]

    # Calculate the midpoint between the first and last non-white rows
    midpoint = (first_non_white_row + last_non_white_row) // 2

    # Blur the grayscale image using a Gaussian filter
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Create a binary mask for pixels with intensities between 0 and 130
    mask = cv2.inRange(blurred, 0, 130)

    # Edge detection by thresholding the blurred image
    _, edges = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    # Combine the mask with edges
    combined = cv2.bitwise_and(edges, edges, mask = mask)

    # Get left side length and right side length
    left_length = np.sum(combined[:, :midpoint] > 0)
    right_length = np.sum(combined[:, midpoint:] > 0)

    # Classify the image
    if left_length > 1.5 * right_length:
        class_name = "real"
    else:
        class_name = "fake"

    return class_name
