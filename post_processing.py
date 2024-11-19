import cv2
import numpy as np
import os

def post_process_mask(predicted_mask, threshold=0.5, kernel_size=(5, 5)):
    _, binary_mask = cv2.threshold(predicted_mask, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones(kernel_size, np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cleaned_mask

def post_process_mask_2(predicted_mask, threshold=0.5, kernel_size=(5, 5), remove_small_objects=True, fill_holes=True):
    _, binary_mask = cv2.threshold(predicted_mask, threshold, 1, cv2.THRESH_BINARY)
    kernel = np.ones(kernel_size, np.uint8)
    opened_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if remove_small_objects:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened_mask, connectivity=8)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip the background label 0
        opened_mask = np.where(labels == largest_label, 1, 0).astype(np.uint8)
    if fill_holes:
        contours, hierarchy = cv2.findContours(opened_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(opened_mask, [cnt], 0, 255, -1)
    if fill_holes:
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    else:
        closed_mask = opened_mask

    return closed_mask * 255 

def save_combined_image(index, actual_image, actual_mask, predicted_mask, output_folder='/results_images'):

    actual_mask_bgr = cv2.cvtColor(actual_mask * 255, cv2.COLOR_GRAY2BGR)
    predicted_mask_bgr = cv2.cvtColor(predicted_mask * 255, cv2.COLOR_GRAY2BGR)

    height = actual_image.shape[0]
    padding = np.full((height, 10, 3), 255, dtype=np.uint8)  # White padding
    combined_image = np.hstack((actual_image, padding, actual_mask_bgr, padding, predicted_mask_bgr))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = str(index) + '.png'
    output_file_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_file_path, combined_image)
    print(f"Combined image saved at {output_filename}")

def save_combined_image_2(index, actual_image, actual_mask, predicted_mask, heatmap , output_folder='weights_files/results_images'):
 
    actual_mask_bgr = cv2.cvtColor(actual_mask * 255, cv2.COLOR_GRAY2BGR)
    predicted_mask_bgr = cv2.cvtColor(predicted_mask * 255, cv2.COLOR_GRAY2BGR)

    height = actual_image.shape[0]
    padding = np.full((height, 10, 3), 255, dtype=np.uint8)  # White padding
    combined_image = np.hstack((actual_image, padding, actual_mask_bgr, padding, predicted_mask_bgr, padding, heatmap))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = str(index) + '.png'
    output_file_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_file_path, combined_image)
    print(f"Combined image saved at {output_filename}")


def save_predictive_mask(mask, index, save_dir):
    pred_mask_binary = post_process_mask_2(mask)
    pred_mask_binary = np.uint8(pred_mask_binary > 0.5) * 255 
    filename = f'predicted_mask_{index}.png'
    cv2.imwrite(save_dir + filename, pred_mask_binary)
    print("Predicted masks saved successfully.")
