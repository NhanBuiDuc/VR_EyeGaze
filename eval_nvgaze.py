#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main program to invoke the rest of the gaze estimation code

@author: Yu Feng
"""
import argparse
import glob
import os
import sys
import pandas as pd
import deepvog
import edgaze
import numpy as np

import numpy as np

def denormalize_gaze_coordinates(gaze_x_norm, gaze_y_norm, video_shape):
    """
    Denormalize gaze coordinates back to original values based on video frame dimensions.
    
    Parameters:
    gaze_x_norm (np.ndarray): Normalized x components.
    gaze_y_norm (np.ndarray): Normalized y components.
    video_shape (tuple): Tuple containing (HEIGHT, WIDTH) of the video frame.
    
    Returns:
    np.ndarray: Denormalized gaze x components.
    np.ndarray: Denormalized gaze y components.
    """
    height, width = video_shape
    
    # Ensure input types are numpy arrays
    gaze_x_norm = np.array(gaze_x_norm)
    gaze_y_norm = np.array(gaze_y_norm)
    
    # Denormalize gaze_x_norm and gaze_y_norm
    gaze_x = 0.5 * (gaze_x_norm + 1.0) * width
    gaze_y = 0.5 * (gaze_y_norm + 1.0) * height
    
    return gaze_x, gaze_y

def average_angular_error(preds_x, preds_y, targets_x, targets_y):
    """
    Calculate and print the Average Angular Error (AAE) between two numpy arrays of 2D gaze vectors.
    
    Parameters:
    preds_x (np.ndarray): Predicted x components.
    preds_y (np.ndarray): Predicted y components.
    targets_x (np.ndarray): Ground truth x components.
    targets_y (np.ndarray): Ground truth y components.
    
    Returns:
    float: The Average Angular Error.
    """
    # Convert lists to numpy arrays for calculation
    preds = np.array([preds_x, preds_y]).T
    targets = np.array([targets_x, targets_y]).T
    
    # Calculate the dot products
    dot_products = np.sum(preds * targets, axis=1)
    
    # Calculate the magnitudes of the vectors
    preds_magnitudes = np.linalg.norm(preds, axis=1)
    targets_magnitudes = np.linalg.norm(targets, axis=1)
    
    # Calculate the cosine of the angle between each pair of vectors
    cos_angles = dot_products / (preds_magnitudes * targets_magnitudes)
    
    # Clip values to ensure they fall within the valid range for arccos
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    
    # Calculate the angular errors in radians
    angular_errors = np.arccos(cos_angles)
    
    # Convert radians to degrees
    angular_errors_degrees = np.degrees(angular_errors)
    
    # Calculate the Average Angular Error
    average_angular_error = np.mean(angular_errors_degrees)
    
    return average_angular_error


def option():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="nvgaze",
        help="path to openEDS dataset, default: openEDS",
    )

    # camera setting
    parser.add_argument(
        "--video_shape",
        nargs=2,
        type=int,
        default=[480, 640],
        help="video_shape, [HEIGHT, WIDTH] default: [400, 640]",
    )
    parser.add_argument(
        "--sensor_size",
        nargs=2,
        type=float,
        default=[3.6, 4.8],
        help="sensor_shape, [HEIGHT, WIDTH] default: [3.6, 4.8]",
    )
    parser.add_argument(
        "--focal_length", type=int, default=6, help="camera focal length, default: 6"
    )


    parser.add_argument(
        "--subject_number",
        type=int,
        default=1,
        help="subject number, default: 1",
    )
    parser.add_argument(
        "--eye",
        type=str,
        default="L",
        help="subject number, default: L",
    )
    # output path
    parser.add_argument("--output_path", help="save folder name", default="output_demo")
    parser.add_argument("--prefix", help="save folder name", default="NV_VR")

    # parse
    args = parser.parse_args()

    return args


def make_dirs(prefix, suffix):
    video_dir = "%s/videos" % prefix
    os.makedirs(video_dir, exist_ok=True)
    eye_model_dir = "%s/fit_models" % prefix
    os.makedirs(eye_model_dir, exist_ok=True)
    result_dir = "%s/results%s" % (prefix, suffix)
    os.makedirs(result_dir, exist_ok=True)

def main():
    args = option()
    output_path = args.output_path
    video_shape = args.video_shape
    prefix = args.prefix
    dataset_path = args.dataset_path
    eye = args.eye
    subject_number = args.subject_number
    path_to_result = os.path.join(output_path, "results", "org_result_" + prefix + "_" + str(subject_number) + "_" + eye + ".csv" )
    path_to_gt = os.path.join(dataset_path,  str(subject_number).zfill(2) + ".csv")

    result_df = pd.read_csv(path_to_result)
    gt_df = pd.read_csv(path_to_gt, comment='#')
    # Filter the data for rows where 'eye' column is 'L'
    filtered_df = gt_df[gt_df['eye'] == eye]

    label_gaze_x = filtered_df['gaze_x'].tolist()
    label_gaze_y = filtered_df['gaze_y'].tolist()

    # Assuming label_gaze_x and label_gaze_y are already defined and normalized
    label_gaze_x, label_gaze_y = denormalize_gaze_coordinates(label_gaze_x, label_gaze_y, video_shape)

    # Split the filtered data into separate arrays
    # Extract the required columns into lists
    pupil2D_x = result_df['pupil2D_x'].tolist()
    pupil2D_y = result_df['pupil2D_y'].tolist()
    gaze_x = result_df['gaze_x'].tolist()
    gaze_y = result_df['gaze_y'].tolist()

    # Calculate angular errors
    pupil2D_error = average_angular_error(pupil2D_x, pupil2D_y, label_gaze_x, label_gaze_y)
    gaze_error = average_angular_error(gaze_x, gaze_y, label_gaze_x, label_gaze_y)
    
    # Print errors
    print(f"Average Angular Error for pupil2D: {pupil2D_error:.2f} degrees")
    print(f"Average Angular Error for gaze: {gaze_error:.2f} degrees")
if __name__ == "__main__":
    main()