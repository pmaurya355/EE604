import cv2
import numpy as np

def largest_contour(mask, image_shape):
    contours, _ = cv2.findContours(image = mask, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)

    # Find the largest contour
    largest_contour = max(contours, key = cv2.contourArea)

    # Draw the largest contour on a blank image
    mask = np.zeros(image_shape, dtype = np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, [255, 255, 255], cv2.FILLED, 1)

    return mask

def modify_mask(mask, image):
    mask_img = image.copy()
    mask_img[mask == [0, 0, 0]] = 0
    hue, _, value = cv2.split(cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV))
    hue = np.mod(hue, 180).astype(np.uint8)
    value = np.mod(value, 180).astype(np.uint8)

    # Adjust the mask based on mean hue and value
    if hue.mean() > 7 or value.mean() < 20: 
        mask = 255 - mask

    return mask

# Usage
def solution(image_path):
    image = cv2.imread(image_path)
    ######################################################################
    ######################################################################
    '''
    The pixel values of output should be 0 and 255 and not 0 and 1
    '''
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################

    # Convert the image to the LAB color space
    modify_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Extract the A and B channels
    _, A_channel, B_channel = cv2.split(modify_image)

    # Stack A and B channels to create a two-dimensional feature matrix
    features = np.float32(np.column_stack((A_channel.ravel(), B_channel.ravel())))

    # Perform K-means clustering
    n = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(features, n, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape labels to the shape of the original image
    segmented_image = labels.reshape(image.shape[:2])

    # Identify the lava cluster
    unique_clusters, counts = np.unique(segmented_image, return_counts = True)
    lava_cluster = unique_clusters[np.argmax(counts)]

    # Create a binary mask for the lava cluster
    lava_mask = (segmented_image != lava_cluster).astype(np.uint8) * 255

    # Find and draw the largest contour on a blank image
    contour_mask = largest_contour(lava_mask, image.shape)
    
    # Modify the mask based on hue and value
    output_mask = modify_mask(contour_mask, image)

    ######################################################################  
    return output_mask

