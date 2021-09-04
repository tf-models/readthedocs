<p align="center">
    <img src="https://raw.githubusercontent.com/tf-models/readthedocs/main/official/projects/yolo3/assets/project_logo.png">
</p>

# Welcome to TensorFlow Model Garden Project Example - YOLOv3 :sunglasses:

[![YOLOv3](http://img.shields.io/badge/Paper-arXiv.1804.02767-B3181B?logo=arXiv)](https://arxiv.org/abs/1804.02767)
[![Python 3.7](https://img.shields.io/badge/Python-3.7-3776AB)](https://www.python.org/downloads/release/python-379/)
[![TensorFlow 2.4](https://img.shields.io/badge/TensorFlow-2.4-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.4.0)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tf-models/readthedocs/blob/main/TFMG_Project_Tutorial_(v6).ipynb)


Implement YOLOv3 by following TensorFlow Model Garden (TFMG) components. Support image and video detection tasks. 

<p align="center">
    <img src="https://raw.githubusercontent.com/tf-models/readthedocs/main/official/projects/yolo3/assets/video_times_square.gif">
</p>

## TFMG Tutorial Colab and Implement Details


<a href="https://githubtocolab.com/tf-models/readthedocs/blob/main/TFMG_Project_Tutorial_(v6).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

### TFMG Components

| Folders      | Required | Description             |
|-------------|----------|-------------------------------------------|
| modeling | yes      | Darknet-53 architecture, YOLO prediction base blocks and models.     |
| ops      | yes      | Operations for bounding boxes and non-maximum suppression (NMS).    |
| losses      | yes      | Loss functions.    |
| dataloaders | yes      | Decoders and parsers for your data pipeline; functions for downloading pretrained weigths.     |
| tasks       | yes      | Tasks for running the model. Tasks are essentially the main driver for training and evaluating the model.     |
| configs     | yes      | The  config  files for the task class to train and evaluate the model.      |
| common      | yes      | Registry imports. The tasks and configs need to be registered before execution.     |
| utils       | no       | Utility functions for draw boxes and video frame interface. |
| demos       | no       | Files needed to create a Jupyter Notebook/Google Colab demo of the model. |

### YOLOv3 Model

![]()

### Useful Tutorials:

- TensorFlow / Keras Guide - Custom Layers and Models: 
    - https://keras.io/api/
    - https://www.tensorflow.org/guide/keras/custom_layers_and_models
- TensorFlow Core API document: 
    - https://www.tensorflow.org/api_docs/python/tf
- A Concise Handbook of TensorFlow 2: 
    - https://tf.wiki/en/


# Installation

## Requirements

- Python 3.7+
- OpenCV 

```
pip install -r requirements.txt
```

## TensorFlow2 Model Garden 

- TensorFlow 2.0+
- Model Garden
- TensorFlow Datasets


```
pip install -q tf-models-nightly tfds-nightly
```

## Download Pretrained Weights

Download pretrained weights from author's official source: 

- Darknet GitHub: https://github.com/pjreddie/darknet
- Joseph Chet Redmon's website: https://pjreddie.com

```
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O ./data/yolov3.weights
```


# Detection Task

Support image detection and video/webcamera detection. 

## Image File

```
python detect.py --image ./data/dog.jpg 
```

<p align="center">
    <img src="outputs/output_dog.jpg">
</p>

## Video File 

```
python detect_video.py --video_path ./data/times_square.mp4 --video True
```

<p align="center">
    <img src="https://raw.githubusercontent.com/tf-models/readthedocs/main/official/projects/yolo3/assets/video_times_square.gif">
</p>


```
find . -size +50M | cat >> .gitignore
```

### Detection

```
# yolov3
python detect.py --image ./data/dog.jpg  
python detect.py --image ./data/street.jpg  
python detect.py --image ./data/eagle.jpg  
python detect.py --image ./data/giraffe.jpg 
python detect.py --image ./data/girl.png

# webcam
python detect_video.py --video 0

# video file
python detect_video.py --video_path ./data/times_square.mp4 --video True
python detect_video.py --video_path ./data/taryn_elliott.mp4 --video True

# video file with output
python detect_video.py --video path_to_file.mp4 --output ./output.avi
```

# Fine-tuning Task



# Training Task
## Training

```
python train.py --
```

### Command Line Args Reference

```
# convert.py

# detect.py

# video_detect.py

# train.py


```
