# TensorFlow2 Model Garden Project Example - YOLOv3 

## TFMG Components


| Folders      | Required | Description                                                                                                                                                                                   |
|-------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **modeling** | yes      | Model and the building blocks.                                                                                                                                                                |
| **ops**     | yes      | Operations: utility functions used by the data pipeline, loss function and modeling.                                                                                                          |
| **losses**      | yes      | Loss function.                                                                                                                                                                                |
| **dataloaders** | yes      | Decoders and parsers for your data pipeline.                                                                                                                                                  |
| configs     | yes      | The  config  files for the task class to train and evaluate the model.                                                                                                                        |
| tasks       | yes      | Tasks for running the model. Tasks are essentially the main driver for training and evaluating the model.                                                                                     |
| common      | yes      | Registry imports. The tasks and configs need to be registered before execution.                                                                                                             |
| utils       | no       | Utility functions for external resources,  e.g. downloading weights, datasets from external sources, and the test cases for these functions. |
| demos       | no       | Files needed to create a Jupyter Notebook/Google Colab demo of the model. |


## Usage
### Installation

```
pip install -r requirements.txt
```

### Download Pretrain 

```
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf
```

### Detection

```
# yolov3
python detect.py --image ./data/meme.jpg

# webcam
python detect_video.py --video 0

# video file
python detect_video.py --video path_to_file.mp4 --weights ./checkpoints/yolov3-tiny.tf --tiny

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

