import skimage.io as io
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial import distance

# Finding the center of a segmented region of an irregular shape can be achieved through various image processing techniques
def irregular_shape_Centroid_radius(segmented_image):
    # Find contours in the segmented image
    contours, _ = cv2.findContours(np.uint8(segmented_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Select the largest contour (assuming it corresponds to the irregular shape)
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])

    distances = []
    for point in largest_contour:
        distance = np.linalg.norm(point[0] - [centroid_x, centroid_y])
        distances.append(distance)

    centroid_radius = np.mean(distances)
    centroid_radius = {"centroid_radius": centroid_radius}
    return centroid_radius

################Mammograms Images ###############
def extract_features_from_mask(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    max_contour = contours[max_contour_index]

    # Contour Approximation
    epsilon = 0.01 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)  
    # Convex Hull
    hull = cv2.convexHull(max_contour)
    
    # Calculate mean x and y values for the approximation
    mean_x_approx = np.mean(approx[:, 0, 0]) if len(approx) > 0 else 0
    mean_y_approx = np.mean(approx[:, 0, 1]) if len(approx) > 0 else 0
    
    # Calculate mean x and y values for the convex hull
    mean_x_hull = np.mean(hull[:, 0, 0])
    mean_y_hull = np.mean(hull[:, 0, 1])
    
    # Calculate std x and y values for the approximation
    std_x_approx = np.std(approx[:, 0, 0]) if len(approx) > 1 else 0
    std_y_approx = np.std(approx[:, 0, 1]) if len(approx) > 1 else 0
    
    # Calculate std x and y values for the convex hull
    std_x_hull = np.std(hull[:, 0, 0]) if len(hull) > 1 else 0
    std_y_hull = np.std(hull[:, 0, 1]) if len(hull) > 1 else 0
    
    features = {
        # 'approximation': approx,
        # 'convex_hull': hull,
        'mean_x_approx': mean_x_approx,
        'mean_y_approx': mean_y_approx,
        'mean_x_hull': mean_x_hull,
        'mean_y_hull': mean_y_hull,
        'std_x_approx': std_x_approx,
        'std_y_approx': std_y_approx,
        'std_x_hull': std_x_hull,
        'std_y_hull': std_y_hull
    }
    return features

def updated_shape_description(segmented_image):
    ##Here segmented region must be a binary image.
    _, binary_mask = cv2.threshold(np.uint8(segmented_image), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask_int = binary_mask.astype(np.uint8)
    props = measure.regionprops(binary_mask_int)
    
    # Calculate width and height of the bounding box
    min_row, min_col, max_row, max_col = props[0].bbox
    width = max_col - min_col
    height = max_row - min_row

    # Circularity
    # The circularity is computed based on the perimeter and area of the region.
    perimeter = props[0].perimeter
    area = props[0].area
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # Normalize area and perimeter with width and height
    normalized_area = area / (width * height)
    normalized_perimeter = perimeter / (width + height)

    # Print or use the shape descriptors as needed
    #print("Total length of Region Boundary:", perimeter )
    #print("Total area of Segmented Image:", area)
    #print("Circularity:", circularity)
    shape_dict = {
        "perimeter": normalized_perimeter,
        "area": normalized_area,
        "circularity": circularity
        # "eccentricity": eccentricity,
        # "solidity": solidity,
        # "convexity": convexity
        }

    return shape_dict

def get_seg_area_pixels(input_img, ground_img):
    # Get the dimensions of the images
    height1, width1 = input_img.shape
    height2, width2 = ground_img.shape
    # Compare the dimensions
    if width1 == width2 and height1 == height2:
        # Extract pixels based on segmentation
        segmented_pixels = input_img[np.where(ground_img == 3)]
        print("Total Segmentated Pixel:", len(segmented_pixels))
        return segmented_pixels
    else:
        return "Error: The Diminsion of the Input Image and Ground Image is Not same"

##Distance form the Center to boundary
def calculate_distances(center_point, boundary_points):
    distances = []
    for point in boundary_points:
        distance = np.linalg.norm(center_point - point)  # Euclidean distance
        distances.append(distance)
    return distances

def distanceMap_Center(segmented_image):
    # Find contours in the segmented image
    contours, _ = cv2.findContours(np.uint8(segmented_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    segmented_contour = max(contours, key=cv2.contourArea)

    # Create a binary mask of the segmented region
    mask = np.zeros(segmented_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [segmented_contour], -1, 255, thickness=cv2.FILLED)
    
  # Get the coordinates of all the points in the segmented region
    segmented_points = np.argwhere(mask == 255)
    segmented_points = np.flip(segmented_points, axis=1) ##flip to the write axis
    distances = distance.cdist(segmented_points, segmented_contour.squeeze(),  metric='euclidean')

    # Create an empty distance map
    distance_min = np.zeros(segmented_image.shape[:2])
    for point, dist in zip(segmented_points, distances):
        distance_min[point[0], point[1]] = dist.min()

    ##find the center points
    p = np.argwhere(distance_min == np.max(distance_min))
   # print("Total Number of Center Points: [Maximum Distance to Counter Points] == ", len(p))

    ##Calculating the distances from the center points
    center_point = p[0]
    boundary_points = segmented_contour.squeeze()

    dists = calculate_distances(center_point, boundary_points)
    min_distance = min(dists)
    max_distance = max(dists)
    C_distance = {
        "min_distance": min_distance,
        "max_distance": max_distance}
    return C_distance
