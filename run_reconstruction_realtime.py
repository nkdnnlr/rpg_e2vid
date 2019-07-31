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

from pyaer import libcaer
from pyaer.davis import DAVIS


def get_event(device):
    data = device.get_event()

    return data




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                        help='path to model weights')
    parser.add_argument('-i', '--input_file', required=True, type=str)
    parser.add_argument('--width', default=240, type=int)
    parser.add_argument('--height', default=180, type=int)
    parser.add_argument('-N', '--window_size', default=None, type=int)
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    set_inference_options(parser)

    args = parser.parse_args()


    # Setting up DAVIS
    ne_device = DAVIS(noise_filter=True)

    print("Device ID:", ne_device.device_id)
    if ne_device.device_is_master:
        print("Device is master.")
    else:
        print("Device is slave.")
    print("Device Serial Number:", ne_device.device_serial_number)
    print("Device String:", ne_device.device_string)
    print("Device USB bus Number:", ne_device.device_usb_bus_number)
    print("Device USB device address:", ne_device.device_usb_device_address)
    print("Device size X:", ne_device.dvs_size_X)
    print("Device size Y:", ne_device.dvs_size_Y)
    print("Logic Version:", ne_device.logic_version)
    print("Background Activity Filter:",
          ne_device.dvs_has_background_activity_filter)

    ne_device.start_data_stream()
    # setting bias after data stream started
    ne_device.set_bias_from_json("./configs/davis346_config.json")

    clip_value = 3
    histrange = [(0, v) for v in (260, 346)]

    num_packet_before_disable = 1000



    # Loading model
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)

    model = model.to(device)
    model.eval()

    reconstructor = ImageReconstructor(model, args.height, args.width, model.num_bins, args)

    """ Read chunks of events using Pandas """
    path_to_events = args.input_file

    # loop through the events and reconstruct images
    N = args.window_size

    if N is None:
        N = int(args.width * args.height * args.num_events_per_pixel)
        print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
            N, args.num_events_per_pixel))
    else:
        print('Will use {} events per tensor (user-specified)'.format(N))
        mean_num_events_per_pixel = float(N) / float(args.width * args.height)
        if mean_num_events_per_pixel < 0.1:
            print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
                   The reconstruction results might be suboptimal.'.format(N))
        elif mean_num_events_per_pixel > 1.5:
            print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
                   The reconstruction results might be suboptimal.'.format(N))

    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset

    if args.compute_voxel_grid_on_cpu:
        print('Will compute voxel grid on CPU.')

    event_tensor_iterator = pd.read_csv(path_to_events, delim_whitespace=True, header=None, names=['t', 'x', 'y', 'pol'],
                                        dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                        engine='c',
                                        skiprows=start_index, chunksize=N, nrows=None)

    event_batch = np.empty((1,5))
    event_batch_size = 0
    event_batch_size_max = 10000
    iterations = 0

    with Timer('Processing entire dataset'):
        while True:
            event_batch = np.empty((1, 5))

            while True:
                if event_batch_size >= event_batch_size_max:
                    break

                try:
                    iterations += 1
                    data = get_event(ne_device)
                    if data is not None:
                        (pol_events, num_pol_event,
                         special_events, num_special_event,
                         frames_ts, frames, imu_events,
                         num_imu_event) = data

                        if pol_events is not None:
                            event_batch = np.concatenate((event_batch, pol_events))
                            event_batch_size += num_pol_event


                except KeyboardInterrupt:
                    ne_device.shutdown()
                    break

            event_batch = event_batch[1:, 0:4]

            event_tensor_pd = pd.DataFrame(data=event_batch,
                                           columns=['t', 'x', 'y', 'pol'],
                                           )
            event_tensor_pd['t'] *= 1e-6
            event_tensor_pd['x'].astype('int16')
            event_tensor_pd['y'].astype('int16')
            event_tensor_pd['pol'].astype('int16')


            last_timestamp = event_tensor_pd.values[-1, 0]

            with Timer('Building event tensor'):
                if args.compute_voxel_grid_on_cpu:
                    event_tensor = events_to_voxel_grid(event_tensor_pd.values,
                                                        num_bins=model.num_bins,
                                                        width=args.width,
                                                        height=args.height)
                    event_tensor = torch.from_numpy(event_tensor)
                else:
                    event_tensor = events_to_voxel_grid_pytorch(event_tensor_pd.values,
                                                                num_bins=model.num_bins,
                                                                width=args.width,
                                                                height=args.height,
                                                                device=device)

            reconstructor.update_reconstruction(event_tensor, start_index + event_batch_size, last_timestamp)

            start_index += event_batch_size
            event_batch_size = 0