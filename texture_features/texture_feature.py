import skimage.io as io
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, skif
from scipy.spatial import distance
from scipy.stats import kurtosis, skew
import shutil

def extract_kinetic_features(roi):
    mean_intensity = np.mean(roi)
    variance = np.var(roi)
    max_intensity = np.max(roi)
    min_intensity = np.min(roi)
    skewness = skew(roi.flatten())
    kurt = kurtosis(roi.flatten())
    entropy = -np.sum(roi * np.log2(roi + 1e-10))
    return {
        "Mean Intensity": mean_intensity,
        "Variance": variance,
        "Max Intensity": max_intensity,
        "Min Intensity": min_intensity,
        "Skewness": skewness,
        "Kurtosis": kurt,
        "Entropy": entropy
    }

# GLCM
def extract_glcm_features(roi):
    glcm = skif.graycomatrix(roi, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = skif.graycoprops(glcm, 'contrast').mean()
    correlation = skif.graycoprops(glcm, 'correlation').mean()
    energy = skif.graycoprops(glcm, 'energy').mean()
    homogeneity = skif.graycoprops(glcm, 'homogeneity').mean()
    return {
        "Contrast": contrast,
        "Correlation": correlation,
        "Energy": energy,
        "Homogeneity": homogeneity
    }

# Function to extract texture features using LBP
def extract_lbp_features(roi):
    lbp = local_binary_pattern(roi, P=8, R=1, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return {f"LBP_{i}": hist[i] for i in range(n_bins)}


################Mammograms Images ###############
def extract_features_from_mask(img):
    # Find contours in the mask
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the index of the largest contour
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
    
    # Store the features
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
    # Calculate shape descriptors
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

    # Return shape descriptors as a dictionary
    shape_dict = {
        "perimeter": normalized_perimeter,
        "area": normalized_area,
        "circularity": circularity
        # "eccentricity": eccentricity,
        # "solidity": solidity,
        # "convexity": convexity
        }

    return shape_dict





########################################################################

###Will return the Input Image area of Left Ventricular, Can change to other classes by updating index number
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

def intensity_statistics(input_image, ground_image):

    intensity_values = input_image[np.where(ground_image)]   ##Index 3 for Left_venticluar Class
    #print(len(intensity_values))
    mean_intensity = np.mean(intensity_values)
    median_intensity = np.median(intensity_values)
    std_intensity = np.std(intensity_values)
    min_intensity = np.min(intensity_values)
    max_intensity = np.max(intensity_values)

    #print("Mean Intensity:", mean_intensity)
    #print("Median Intensity:", median_intensity)
    #print("STD Intensity:", std_intensity)
    #print("Minimum Intensity:", min_intensity)
    #print("Maximum Intensity:", max_intensity)

    # Return shape descriptors as a dictionary
    intensity_dict = {
        "Mean": mean_intensity,
        "Median": median_intensity,
        "STD": std_intensity,
        "Min_Intensity": min_intensity,
        "Max_Intensity": max_intensity }

    return intensity_dict


def shape_description(segmented_image):
    # Calculate shape descriptors
    ##Here segmented region must be a binary image.
    _, binary_mask = cv2.threshold(np.uint8(segmented_image), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask_int = binary_mask.astype(np.uint8)
    props = measure.regionprops(binary_mask_int)

    # Circularity
    # The circularity is computed based on the perimeter and area of the region.
    perimeter = props[0].perimeter
    area = props[0].area
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # Eccentricity represents the elongation of the region,
    ##with values closer to 0 indicating a more circular shape.
    eccentricity = props[0].eccentricity

    # Solidity measures the ratio of the region's area to its convex hull's area.
    solidity = props[0].solidity

    # Convexity is calculated by dividing the area of the region by the area of its convex hull.
    convex_area = props[0].convex_area
    convexity = area / convex_area

    # Print or use the shape descriptors as needed
    #print("Total length of Region Boundary:", perimeter )
    #print("Total area of Segmented Image:", area)
    #print("Circularity:", circularity)
    #print("Eccentricity:", eccentricity)
    #print("Solidity:", solidity)
    #print("Convexity:", convexity)

    # Return shape descriptors as a dictionary
    shape_dict = {
        "perimeter": perimeter,
        "area": area,
        "circularity": circularity
        # "eccentricity": eccentricity,
        # "solidity": solidity,
        # "convexity": convexity
        }

    return shape_dict


def texture_feature(segmented_image):

    grayco_M = segmented_image.astype(np.uint8)

    levels = grayco_M.max() + 1
    glcm = graycomatrix(grayco_M, distances=[5], angles=[0], levels=levels, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    #print("Contrast:", contrast)
    #print("Homogeneity:", homogeneity)

    # Return shape descriptors as a dictionary
    texture_dict = {
        "contrast": contrast,
        "Homogeneity": homogeneity
        }

    return texture_dict



# Finding the center of a segmented region of an irregular shape can be achieved through various image processing techniques
def irregular_shape_Centroid_radius(segmented_image):

    # Compute the region properties for the labeled regions:
    # Find contours in the segmented image
    contours, _ = cv2.findContours(np.uint8(segmented_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Select the largest contour (assuming it corresponds to the irregular shape)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the centroid of the contour
    # Moments are statistical measures used to describe the shape and spatial distribution of an object in an image.
    # The first-order moments,specifically the centroid or center of mass, can provide the coordinates
    # of the center of the segmented region
    M = cv2.moments(largest_contour)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])

    # Calculate the average distance between the centroid and the contour points
    distances = []
    for point in largest_contour:
        distance = np.linalg.norm(point[0] - [centroid_x, centroid_y])
        distances.append(distance)

    #radius is the mean of all the distances to boundry points
    centroid_radius = np.mean(distances)

    centroid_radius = {"centroid_radius": centroid_radius}

    return centroid_radius

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
    #print("Total area of Binary mask", segmented_points.shape[0])
    segmented_points = np.flip(segmented_points, axis=1) ##flip to the write axis

    # Calculate the pairwise distances from each point to the contour points
    distances = distance.cdist(segmented_points, segmented_contour.squeeze(),  metric='euclidean')

    # Create an empty distance map
    distance_min = np.zeros(segmented_image.shape[:2])

    # Assign the distances to the corresponding pixel positions in the distance map
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

    #print("Minimum distance:", min_distance)
    #print("Maximum distance:", max_distance)

    # Return shape descriptors as a dictionary
    C_distance = {
        "min_distance": min_distance,
        "max_distance": max_distance}

    return C_distance



