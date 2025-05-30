import json
import argparse
import time
from os.path import join, basename

import numpy as np
import torch
import pandas as pd

from utils.loading_utils import load_model, get_device
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluating a trained network")
    parser.add_argument(
        "-c", "--path_to_model", required=True, type=str, help="path to model weights"
    )
    parser.add_argument("-i", "--input_file", required=True, type=str)
    parser.add_argument("--width", default=240, type=int)
    parser.add_argument("--height", default=180, type=int)
    parser.add_argument("-N", "--window_size", default=None, type=int)
    parser.add_argument(
        "--num_events_per_pixel",
        default=0.35,
        type=float,
        help="in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel",
    )
    parser.add_argument("--skipevents", default=0, type=int)
    parser.add_argument("--suboffset", default=0, type=int)
    parser.add_argument(
        "--compute_voxel_grid_on_cpu",
        dest="compute_voxel_grid_on_cpu",
        action="store_true",
    )
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    set_inference_options(parser)

    args = parser.parse_args()

    # Loading model
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)

    model = model.to(device)
    model.eval()

    reconstructor = ImageReconstructor(
        model, args.height, args.width, model.num_bins, args
    )

    """ Read chunks of events using Pandas """
    path_to_events = args.input_file

    # loop through the events and reconstruct images
    N = args.window_size

    if N is None:
        N = int(args.width * args.height * args.num_events_per_pixel)
        print(
            "Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).".format(
                N, args.num_events_per_pixel
            )
        )
    else:
        print("Will use {} events per tensor (user-specified)".format(N))
        mean_num_events_per_pixel = float(N) / float(args.width * args.height)
        if mean_num_events_per_pixel < 0.1:
            print(
                "!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
                   The reconstruction results might be suboptimal.".format(
                    N
                )
            )
        elif mean_num_events_per_pixel > 1.5:
            print(
                "!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
                   The reconstruction results might be suboptimal.".format(
                    N
                )
            )

    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset

    if args.compute_voxel_grid_on_cpu:
        print("Will compute voxel grid on CPU.")

    event_tensor_iterator = pd.read_csv(
        path_to_events,
        delim_whitespace=True,
        header=None,
        names=["t", "x", "y", "pol"],
        dtype={"t": np.float64, "x": np.int16, "y": np.int16, "pol": np.int16},
        engine="c",
        skiprows=start_index,
        chunksize=N,
        nrows=None,
    )

    with Timer("Processing entire dataset"):
        for event_tensor_pd in event_tensor_iterator:

            last_timestamp = event_tensor_pd.values[-1, 0]

            with Timer("Building event tensor"):
                if args.compute_voxel_grid_on_cpu:
                    event_tensor = events_to_voxel_grid(
                        event_tensor_pd.values,
                        num_bins=model.num_bins,
                        width=args.width,
                        height=args.height,
                    )
                    event_tensor = torch.from_numpy(event_tensor)
                else:
                    event_tensor = events_to_voxel_grid_pytorch(
                        event_tensor_pd.values,
                        num_bins=model.num_bins,
                        width=args.width,
                        height=args.height,
                        device=device,
                    )

            reconstructor.update_reconstruction(
                event_tensor, start_index + N, last_timestamp
            )

            start_index += N
