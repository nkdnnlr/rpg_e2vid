# High Speed and High Dynamic Range Video with an Event Camera

[![High Speed and High Dynamic Range Video with an Event Camera](http://rpg.ifi.uzh.ch/E2VID/video_thumbnail.png)](https://youtu.be/eomALySSGVU)

This is the code for the paper **High Speed and High Dynamic Range Video with an Event Camera** by [Henri Rebecq](http://henri.rebecq.fr), Rene Ranftl, [Vladlen Koltun](http://vladlen.info/) and [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html):

You can find a pdf of the paper [here](http://rpg.ifi.uzh.ch/docs/arXiv19_Rebecq.pdf).
If you use any of this code, please cite the following publications:

```bibtex
@Article{Rebecq19arxiv,
  author        = {Henri Rebecq and Ren{\'{e}} Ranftl and Vladlen Koltun and Davide Scaramuzza},
  title         = {High Speed and High Dynamic Range Video with an Event Camera},
  journal       = {ar{X}iv e-prints},
  url           = {https://arxiv.org/abs/1906.07165}
  year          = 2019
}
```


```bibtex
@Article{Rebecq19cvpr,
  author        = {Henri Rebecq and Ren{\'{e}} Ranftl and Vladlen Koltun and Davide Scaramuzza},
  title         = {Events-to-Video: Bringing Modern Computer Vision to Event Cameras},
  journal       = {{IEEE} Conf. Comput. Vis. Pattern Recog. (CVPR)},
  year          = 2019
}
```

## Install

Dependencies:

- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.0
- [NumPy](https://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [OpenCV](https://opencv.org/)

### Install with Anaconda

The installation requires [Anaconda3](https://www.anaconda.com/distribution/).
You can create a new Anaconda environment with the required dependencies as follows (make sure to adapt the CUDA toolkit version according to your setup):

```bash
conda create -n E2VID
conda activate E2VID
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install pandas
conda install -c conda-forge opencv
```

## Run

- Download the pretrained model:

```bash
mkdir -p pretrained
wget "https://seafile.ifi.uzh.ch/f/457050be5aef4ed59ef6/?dl=1" -O pretrained/E2VID.pth.tar
```

- Download an example file with event data:

```bash
mkdir -p data
wget "https://seafile.ifi.uzh.ch/f/ac6058453cc34ff5ab82/?dl=1" -O data/dynamic_6dof.zip
```

Before running the reconstruction, make sure the conda environment is sourced:

```bash
conda activate E2VID
```

- Run reconstruction:

```bash
python run_reconstruction.py \
  -c pretrained/E2VID.pth.tar \
  -i data/dynamic_6dof.zip \
  --auto_hdr \
  --display \
  --show_events
```

## Parameters

Below is a description of the most important parameters:

#### Main parameters

- ``--window_size`` / ``-N`` (default: None) Number of events per tensor. This is the parameter that has the most influence of the image reconstruction quality. If set to None, this number will be automatically computed based on the sensor size, as N = width * height * num_events_per_pixel (see description of that parameter below).
- ``--auto_hdr`` (default: True) If True, the output image values (in the range [0,1]) will be rescaled using (robust) min/max normalization. See also the parameters ``--auto_hdr_min_percentile``, ``--auto_hdr_max_percentile``, and ``--auto_hdr_moving_average_size``.
- ``--safety_margin`` (default: 5): zero-pad the input event tensors with zeros to avoid boundary effects (typical range: [0,5]). A small value reduces computation time but may introduce boundary effects.
- ``--hot_pixels_file`` (default: None): Path to a file specifying the locations of hot pixels (such a file can be obtained with [this tool](https://github.com/cedric-scheerlinck/dvs_tools/tree/master/dvs_hot_pixel_filter) for example). These pixels will be ignored (i.e. zeroed out in the event tensors).
- ``--color`` (default: False): if True, will perform color reconstruction as described in the paper. Only use this with a [color event camera](http://rpg.ifi.uzh.ch/CED.html) such as the Color DAVIS346. When ``--color`` is set, the ``--auto_hdr`` flag will be automatically set. Do not forget to set ``width=346`` and ``height=260`` for the ColorDAVIS346.

#### Output parameters

- ``--output_folder``: path of the output folder. If not set, the image reconstructions will not be saved to disk.
- ``--dataset_name``: name of the output folder directory (default: 'reconstruction').

#### Display parameters

- ``--display`` (default: False): display the video reconstruction in real-time in an OpenCV window.
- ``--show_events`` (default: False): show the input events side-by-side with the reconstruction. If ``--output_folder`` is set, the previews will also be saved to disk in ``/path/to/output/folder/events``.

#### Additional parameters

- ``--num_events_per_pixel`` (default: 0.35): Parameter used to automatically estimate the window size based on the sensor size. The value of 0.35 was chosen to correspond to ~ 15,000 events on a 240x180 sensor such as the DAVIS240C.
- ``--no-normalize`` (default: False): Disable event tensor normalization: this will improve speed a bit, but might degrade the image quality a bit.
- ``--no-recurrent`` (default: False): Disable the recurrent connection (i.e. do not maintain a state). For experimenting only, the results will be flickering a lot.

## Example datasets

We provide a list of example (publicly available) event datasets to get started with E2VID.

- [Event Camera Dataset](https://seafile.ifi.uzh.ch/d/31a928bb230e4f8dbef4/)
- [Bardow et al., CVPR'16](https://seafile.ifi.uzh.ch/d/ac21d1dd21db443eb165/)
- [Scherlinck et al., ACCV'18](https://seafile.ifi.uzh.ch/d/b59ad45811674ac5a49f/)
- [Color event sequences from the CED dataset Scheerlinck et al., CVPR'18](https://seafile.ifi.uzh.ch/d/fe294093da7f46d2867c/)

## Working with ROS

Because PyTorch recommends Python 3 and ROS is only compatible with Python2, it is not straightforward to have the PyTorch reconstruction code and ROS code running in the same environment.
To make things easy, the reconstruction code we provide has no dependency on ROS, and simply read events from a text file or ZIP file.
We provide convenience functions to convert ROS bags (a popular format for event datasets) into event text files.
In addition, we also provide scripts to convert a folder containing image reconstructions back to a rosbag (or to append image reconstructions to an existing rosbag).

**Note**: it is **not** necessary to have a sourced conda environment to run the following scripts. However, [ROS](https://www.ros.org/) needs to be installed and sourced.

### rosbag -> events.txt

To extract the events from a rosbag to a zip file containing the event data:

```bash
python scripts/extract_events_from_rosbag.py /path/to/rosbag.bag \
  --output_folder=/path/to/output/folder \
  --event_topic=/dvs/events
```

### image reconstruction folder -> rosbag

```bash
python scripts/image_folder_to_rosbag.py \
  --rosbag_folder /path/to/rosbag/folder \
  --datasets dynamic_6dof \
  --image_folder /path/to/image/folder \
  --output_folder /path/to/output_folder \
  --image_topic /dvs/image_reconstructed
```

### Append image_reconstruction_folder to an existing rosbag

```bash
cd scripts
python embed_reconstructed_images_in_rosbag.py \
  --rosbag_folder /path/to/rosbag/folder \
  --datasets dynamic_6dof \
  --image_folder /path/to/image/folder \
  --output_folder /path/to/output_folder\
  --image_topic /dvs/image_reconstructed
```

### Generating a video reconstruction (with a fixed framerate)

It can be convenient to convert an image folder to a video with a fixed framerate (for example for use in a video editing tool).
You can proceed as follows:

```bash
export FRAMERATE=30
python resample_reconstructions.py -i /path/to/input_folder -o /tmp/resampled -r $FRAMERATE
ffmpeg -framerate $FRAMERATE -i /tmp/resampled/frame_%010d.png video_"$FRAMERATE"Hz.mp4
```

## Running Real-Time with PyAER
PyAER is a low-level Python APIs for Accessing Neuromorphic Devices, written by Yuhuang Hu of the Institute of Neuroinformatics Zurich (email: duguyue100@gmail.com). Here we use it to process the event stream in real-time. 
For installation and documentation, see https://dgyblog.com/pyaer-doc/. 

Before running the code, make sure to get configs by e.g. running  `make davis346-test` in the PyAER directory. 

Run code with 

```
python run_reconstruction_realtime.py   -c pretrained/E2VID.pth.tar  --auto_hdr   --display   --show_events --output_folder='output'
```



## Acknowledgements

This code borrows from the following open source projects, whom we would like to thank:

- [pytorch-template](https://github.com/victoresque/pytorch-template)
- [PyAER](https://github.com/duguyue100/pyaer)
