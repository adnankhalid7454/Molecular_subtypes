
import cv2
from albumentations import Compose, Rotate, CLAHE
import numpy as np
import os

def truncation_normalization(img):
    Pmin = np.percentile(img[img!=0], 5)
    Pmax = np.percentile(img[img!=0], 99)
    truncated = np.clip(img,Pmin, Pmax)  
    normalized = (truncated - Pmin)/(Pmax - Pmin)
    normalized[img==0]=0
    return normalized

def rotate_image_and_mask(image, mask, angle):
        center = (image.shape[1]//2, image.shape[0]//2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        rotated_mask = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]))
        return rotated_image, rotated_mask

def clahe_(img, clip):  # clip value is 0.1 or 0.2
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img  # Assume the image is already grayscale
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    cl = clahe.apply(img_gray)
    return cl

def to_rgb(img):
    if len(img.shape) == 2 or img.shape[2] == 1:  # Check if the image is grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def augmentation(img, msk):
    augmented_pairs = []
    for angle in [0, 90, 180, 270]:
        rot_img, rot_msk = rotate_image_and_mask(img, msk, angle)
        augmented_pairs.append((rot_img, rot_msk))

    for clip_limit in [2.0, 4.0]:
        # Define the CLAHE augmentation
        clahe_augmentation = clahe_(img, clip_limit)
        augmented_pairs.append((clahe_augmentation, msk))
    augmented_images, augmented_masks = zip(*augmented_pairs)
    augmented_images_rgb = [to_rgb(img) for img in augmented_images]
    return np.stack(augmented_images_rgb), np.stack(augmented_masks)


def plot_augmented_images_and_masks(images, masks):
    num_samples = images.shape[0]
    plt.figure(figsize=(2 * num_samples, 4))  
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(images[i])
        plt.axis('off') 
        plt.title(f'Image {i+1}')
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(masks[i], cmap='gray') 
        plt.axis('off')  # Turn off axis labels
        plt.title(f'Mask {i+1}') 
    plt.tight_layout()
    plt.show()
# plot_augmented_images_and_masks(augX, augY)
