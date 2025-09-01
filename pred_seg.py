#!/usr/bin/env python3
# -*- coding: utf-8 -*-

Example:
python test_segment_and_save.py \
  --input_dir "/testing/images" \
  --weights "Fine_tune_weights/Seg_model.h5" \
  --out_dir "results" \
  --img_rows 256 --img_cols 256
"""

import os
import glob
import argparse
import numpy as np
import cv2
import tensorflow as tf


# If a GPU is available, let TF allocate memory on demand
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# Model import from your project
# Ensure these modules are importable from your repo root
from C_Unets_Attent import Unets_AttentBEM

# If you need custom losses or metrics only for compile, you can import them.
# Loading weights does not require compile, so we skip it to reduce dependencies.


def load_images_from_dir(input_dir, valid_exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
    paths = []
    for ext in valid_exts:
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    return paths

def apply_histogram_equalization_bgr(img_bgr):
    """
    Your training snippet used grayscale equalization then converted back to BGR.
    We follow the same approach for consistency with training preprocessing.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    bgr_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    return bgr_eq

def preprocess_image_for_model(img_bgr, target_rows, target_cols):
    """
    Resize to model input size, apply histogram equalization, then normalize to [0,1].
    Returns both:
      img_for_model: float32 HxWx3 in [0,1] at target size
      img_resized_bgr: uint8 HxWx3 at target size for visualization overlays
    """
    img_resized = cv2.resize(img_bgr, (target_cols, target_rows), interpolation=cv2.INTER_AREA)
    img_eq = apply_histogram_equalization_bgr(img_resized)
    img_for_model = img_eq.astype("float32") / 255.0
    return img_for_model, img_resized

def post_process_mask_2(predicted_mask, threshold=0.5, kernel_size=(5, 5), remove_small_objects=True, fill_holes=True):
    """
    predicted_mask: 2D float array in [0,1] or raw sigmoid output
    returns uint8 mask in {0,255}
    """
    _, binary_mask = cv2.threshold(predicted_mask, threshold, 1, cv2.THRESH_BINARY)
    kernel = np.ones(kernel_size, np.uint8)

    opened_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    if remove_small_objects:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            opened_mask = np.where(labels == largest_label, 1, 0).astype(np.uint8)

    if fill_holes:
        contours, _ = cv2.findContours(opened_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(opened_mask, [cnt], 0, 255, -1)

        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    else:
        closed_mask = opened_mask

    return (closed_mask > 0).astype(np.uint8) * 255

def make_side_by_side(original_bgr, mask_uint8, alpha=0.35):
    """
    Create a side by side image: left original, right overlay of mask on original.
    """
    # Ensure mask has 3 channels
    mask_color = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 1.0, mask_color, alpha, 0)

    left = original_bgr
    right = overlay
    side_by_side = np.hstack([left, right])
    return side_by_side


def build_model(img_rows, img_cols):
    model = Unets_AttentBEM()
    # Build the model by calling it once with a dummy input
    _ = model(tf.zeros((1, img_rows, img_cols, 3), dtype=tf.float32), training=False)
    return model

def predict_mask(model, img_batch):
    preds = model.predict(img_batch, verbose=0)
    # Model may output [N, H, W, 1], squeeze last dim
    if preds.ndim == 4 and preds.shape[-1] == 1:
        preds = preds[..., 0]
    return preds

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # Build and load weights
    model = build_model(args.img_rows, args.img_cols)
    model.load_weights(args.weights)

    image_paths = load_images_from_dir(args.input_dir)

    batch_imgs = []
    batch_meta = []  # store tuples of (orig_path, orig_shape, resized_bgr)
    saved_count = 0

    def flush_batch():
        nonlocal saved_count, batch_imgs, batch_meta
        if not batch_imgs:
            return
        batch = np.stack(batch_imgs, axis=0)
        preds = predict_mask(model, batch)

        for i in range(preds.shape[0]):
            pred = preds[i]
            orig_path, orig_h, orig_w, resized_bgr = batch_meta[i]

            # Post process at model resolution then resize back to original image size
            post_mask = post_process_mask_2(pred, threshold=args.threshold)
            mask_back = cv2.resize(post_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # Prepare visualization at original resolution
            orig_bgr = cv2.imread(orig_path, cv2.IMREAD_COLOR)
            if orig_bgr is None:
                # fallback to resized if original cannot be read
                orig_bgr = resized_bgr

            vis = make_side_by_side(orig_bgr, mask_back, alpha=args.overlay_alpha)

            base = os.path.splitext(os.path.basename(orig_path))[0]
            mask_path = os.path.join(args.out_dir, f"mask_{base}.png")
            vis_path = os.path.join(args.out_dir, f"vis_{base}.png")

            cv2.imwrite(mask_path, mask_back)
            cv2.imwrite(vis_path, vis)
            saved_count += 2

        # clear batch
        batch_imgs = []
        batch_meta = []

    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: could not read {p}, skipping.")
            continue

        orig_h, orig_w = img.shape[:2]
        img_for_model, resized_bgr = preprocess_image_for_model(img, args.img_rows, args.img_cols)

        batch_imgs.append(img_for_model)
        batch_meta.append((p, orig_h, orig_w, resized_bgr))

        if len(batch_imgs) == args.batch_size:
            flush_batch()

    # flush any remaining
    flush_batch()

    print(f"Done. Saved results to: {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation testing script that saves masks and visualizations for GitHub")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with input patch images")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights .h5")
    parser.add_argument("--out_dir", type=str, default="github_results", help="Output directory")
    parser.add_argument("--img_rows", type=int, default=256, help="Model input rows")
    parser.add_argument("--img_cols", type=int, default=256, help="Model input cols")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for prediction")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing masks")
    parser.add_argument("--overlay_alpha", type=float, default=0.35, help="Alpha for mask overlay in visualization")
    args = parser.parse_args()
    main(args)
