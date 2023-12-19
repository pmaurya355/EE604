import cv2
import numpy as np

def normalize_image(image):
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image)) 
    return normalized_image

def bilateral_filter(image, d, sigma_color, sigma_space):
    height, width, channels = image.shape
    filtered_image = np.zeros_like(image, dtype = np.float32)

    for y in range(height):
        for x in range(width):
            window = image[max(0, y - d) : min(height, y + d + 1), max(0, x - d) : min(width, x + d + 1)]
            intensity_diff = np.exp(-0.5 * np.sum((window - image[y, x])**2, axis = -1) / (sigma_color**2))
            
            # Calculate spatial_diff based on Euclidean distance
            spatial_diff_y = np.exp(-0.5 * ((np.arange(window.shape[0]) - d)**2) / (sigma_space**2))
            spatial_diff_x = np.exp(-0.5 * ((np.arange(window.shape[1]) - d)**2) / (sigma_space**2))
            spatial_diff = np.outer(spatial_diff_y, spatial_diff_x)
            
            weights = intensity_diff * spatial_diff
            normalized_weights = weights / np.sum(weights)

            # Use vector operations for window processing
            filtered_image[y, x, :] = np.sum(window * normalized_weights[:, :, np.newaxis], axis = (0, 1))

    return filtered_image

def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    # image = cv2.imread(image_path_b)

    # Read the images
    images = [cv2.imread(path) for path in [image_path_a, image_path_b]]

    # Convert the images to floating point representation
    images = [image.astype(np.float32) / 255.0 for image in images]

    # Normalize the exposure levels of the images
    norm_images = [normalize_image(image) for image in images]

    # Perform bilateral filtering on each image
    bilateral_images = [bilateral_filter(image, d = 1, sigma_color = 50.0, sigma_space = 0.2) for image in norm_images]

    # Take average of two images
    fused_image = np.zeros_like(images[0])

    for i in range(len(images)):
        fused_image += 0.5 * bilateral_images[i]

    # Convert the fused image back to 8-bit representation
    output_image = (fused_image * 255).clip(0, 255).astype(np.uint8)

    ######################################################################
    return output_image


