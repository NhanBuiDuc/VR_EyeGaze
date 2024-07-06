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

def option():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="nvgaze",
        help="path to openEDS dataset, default: openEDS",
    )
    parser.add_argument(
        "--sequence", type=int, help="The sequence number of the dataset", default=1
    )
    parser.add_argument(
        "--all_sequence",
        default=False,
        action="store_true",
        help="Evaluate all sequences in dataset",
    )
    parser.add_argument(
        "--preview",
        default=False,
        action="store_true",
        help="preview the result, default: False",
    )

    # model information
    parser.add_argument(
        "--model",
        type=str,
        default="eye_net",
        help="dnn model, support eye_net, eye_net_m, pruned_eye_net",
    )
    parser.add_argument(
        "--pytorch_model_path",
        type=str,
        help="pytorch model path",
        default="model_weights/eye_net.pkl",
    )
    parser.add_argument(
        "--disable_edge_info",
        default=False,
        action="store_true",
        help="disable using edge information in bbox prediction",
    )
    parser.add_argument(
        "--use_sift",
        default=False,
        action="store_true",
        help="use sift descriptor to extrapolate bbox",
    )
    parser.add_argument(
        "--bbox_model_path",
        type=str,
        default="model_weights/G030_c32_best.pth",
        help="bbox model path",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device to run on, 'cpu' or 'cuda', only apply to pytorch, default: CPU",
        default="cuda",
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

    # ROI-related settings
    parser.add_argument(
        "--mode",
        type=str,
        default="org",
        help="processing mode, org: use baseline [default], filter: use smart camera filter",
    )
    parser.add_argument(
        "--scaledown",
        type=int,
        default=2,
        help="scaledown when tracking bbox to reduce computation, default: 2",
    )
    parser.add_argument(
        "--blur_size",
        type=int,
        default=3,
        help="blur the input image when tracking bbox",
    )
    parser.add_argument(
        "--clip_val",
        type=int,
        default=10,
        help="clip value in event emulator, clip all low pixel value to this number, default: 10",
    )
    parser.add_argument(
        "--threshold_ratio",
        type=float,
        default=0.3,
        help="threshold ratio to activate an event, default: 0.3",
    )
    parser.add_argument(
        "--density_threshold",
        type=float,
        default=0.05,
        help="threshold ratio to warp result, default: 0.05",
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
        default="R",
        help="subject number, default: L",
    )
    # output path
    parser.add_argument("--output_path", help="save folder name", default="output_demo")
    parser.add_argument("--suffix", help="save folder name", default="")

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

def eval():
    pass
def main():

    args = option()

    if args.mode != "org" and args.mode != "filter" and args.mode != "eval":
        print("[ERROR]: '--mode' only support 'org' and 'filter'.")
        exit()

    # some initial setting
    video_shape = (args.video_shape[0], args.video_shape[1])
    sensor_size = (args.sensor_size[0], args.sensor_size[1])
    focal_length = args.focal_length
    eye = args.eye
    model_path = args.pytorch_model_path
    use_sift = args.use_sift
    disable_edge_info = args.disable_edge_info
    subject_number = args.subject_number
    keyword = args.mode
    result_prefix = args.output_path
    result_suffix = args.suffix
    device = args.device
    # make output directory
    make_dirs(result_prefix, result_suffix)

    # this dataset has 200 sequences of data in total
    seq_start = 0
    path = os.path.join(args.dataset_path, str(subject_number), "/*")
    path = args.dataset_path + "/" + str(subject_number) + "/*"
    seq_end = len(glob.glob(path))

    # if not args.all_sequence:
    #     seq_start = args.sequence
    #     seq_end = args.sequence + 1
        # Load our pre-trained network for baseline
    file_path = os.path.join("nvgaze", str(subject_number).zfill(2) + ".csv")
    # check_and_create_path(file_path)
    # Read the CSV file, skipping the commented lines
    df = pd.read_csv(file_path, comment='#')

    # Filter the data for rows where 'eye' column is 'L'
    filtered_df = df[df['eye'] == eye]

    # Split the filtered data into separate arrays
    imagefiles = filtered_df['imagefile'].to_list()

    log_record_path = "%s/NV_VR/%d/%s/" % (result_prefix, args.subject_number, eye)
    dataset = "%s/%i" % (args.dataset_path, args.subject_number)
    output_record_path = "%s/results%s/%s_result_NV_VR_%d_%s.csv" % (
        result_prefix,
        result_suffix,
        keyword,
        args.subject_number,
        eye
    )
    video_name = "%s/videos/%s_NV_VR_%d_%s.avi" % (result_prefix, keyword, args.subject_number, eye)
    model_name = "eye_fit_models/NV_VR/%d/%s.json" % (args.subject_number, eye)
    model = edgaze.eye_segmentation.EyeSegmentation(
        model_name=args.model,
        model_path=model_path,
        device=args.device,
        preview=args.preview,
    )
    # Init our smart camera filter instance
    filter_model = edgaze.pipeline.Framework(
        model_name=args.model,
        model_path=model_path,
        device=args.device,
        scaledown=args.scaledown,
        record_path=log_record_path,
        use_sift=use_sift,
        disable_edge_info=disable_edge_info,
        bbox_model_path=args.bbox_model_path,
        blur_size=args.blur_size,
        clip_val=args.clip_val,
        threshold_ratio=args.threshold_ratio,
        density_threshold=args.density_threshold,
        preview=args.preview,
    )

    # Initialize the class. It requires information of your camera's focal
    # length and sensor size, which should be available in product manual.
    inferer = deepvog.nvgaze_inferer(
        model, filter_model, focal_length, video_shape, sensor_size, args.subject_number, eye
    )
    for idx, data in enumerate(imagefiles):
        # # Fit an eyeball model.
        inferer.process(dataset, mode="Fit", keyword=keyword, model_name = model_name, data = data, idx = idx)
        print("\r%s (%d/%d)" % (dataset, idx + 1, len(imagefiles)), end="\n", flush=True)
        # # store the eye model
        inferer.save_eyeball_model(model_name)

        # load the eyeball model
        inferer.load_eyeball_model(model_name)

        # infer gaze

        inferer.process(
            dataset,
            mode="Infer",
            keyword=keyword,
            output_record_path=output_record_path,
            output_video_path=video_name,
            data = data,
            idx = idx
        )


if __name__ == "__main__":
    main()