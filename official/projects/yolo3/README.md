<p align="center">
    <img src="https://raw.githubusercontent.com/tf-models/readthedocs/main/official/projects/yolo3/assets/Logo_TFMG_YOLO.png">
</p>

# TensorFlow Model Garden project example - YOLOv3

[![YOLOv3](http://img.shields.io/badge/Paper-arXiv.1804.02767-B3181B?logo=arXiv)](https://arxiv.org/abs/1804.02767)


Implement YOLOv3 following TensorFlow Model Garden (TFMG) components. 

<p align="center">
    <img src="https://raw.githubusercontent.com/tf-models/readthedocs/main/official/projects/yolo3/assets/video_times_square.gif">
</p>

## Implement Detail and TFMG Tutorial Colab
<a href="https://githubtocolab.com/tf-models/readthedocs/blob/main/TFMG_Project_Tutorial_(v6).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>


## Usage

### Requirement

- Python 3.7+
- TensorFlow 2.0+
- OpenCV 
### Installation

```
# install 
pip install -q tf-models-nightly tfds-nightly
pip install -r requirements.txt
```

### Download Pretrain 

```
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf
```

# Detection Demo

Support image detection and video/webcamera detection. 

## Image 

```
python detect.py --image ./data/dog.jpg 
```

<p align="center">
    <img src="outputs/output_dog.jpg">
</p>

## Video 

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

### Training

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

## Implementation 


### Darknet

