<p align="center">
    <img src="https://raw.githubusercontent.com/tf-models/readthedocs/main/official/projects/yolo3/assets/project_logo.png">
</p>

# Welcome to TensorFlow Model Garden project example - YOLOv3 :sunglasses:

[![YOLOv3](http://img.shields.io/badge/Paper-arXiv.1804.02767-B3181B?logo=arXiv)](https://arxiv.org/abs/1804.02767)


Implement YOLOv3 by following TensorFlow Model Garden (TFMG) components. Support image and video detection tasks. 

<p align="center">
    <img src="https://raw.githubusercontent.com/tf-models/readthedocs/main/official/projects/yolo3/assets/video_times_square.gif">
</p>

## Implement Detail and TFMG Tutorial Colab
<a href="https://githubtocolab.com/tf-models/readthedocs/blob/main/TFMG_Project_Tutorial_(v6).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>


| Folders      | Required | Description                                                                                                                                                                                   |
|-------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **modeling** | yes      | Darknet-53 architecture and YOLO prediction models and base blocks.                                                                                                                                                              |
| **ops**     | yes      | Operations: utility functions used by the data pipeline, loss function and modeling.                                                                                                          |
| **losses**      | yes      | Loss                                                                                                                                                                |
| **dataloaders** | yes      | Decoders and parsers for your data pipeline.                                                                                                                                                  |
| configs     | yes      | The  config  files for the task class to train and evaluate the model.                                                                                                                        |
| tasks       | yes      | Tasks for running the model. Tasks are essentially the main driver for training and evaluating the model.                                                                                     |
| common      | yes      | Registry imports. The tasks and configs need to be registered before execution.                                                                                                             |
| utils       | no       | Utility functions for external resources,  e.g. downloading weights, datasets from external sources, and the test cases for these functions. |
| demos       | no       | Files needed to create a Jupyter Notebook/Google Colab demo of the model. |




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
