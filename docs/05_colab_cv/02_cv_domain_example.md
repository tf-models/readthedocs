# TF-Vision Domain-Specific Example

## TF-Vision Tutorial - YOLO (Object Detectors, You Only Look Once) v3

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorflow/models/blob/master/official/vision/beta/projects/movinet/movinet_tutorial.ipynb) -->
<!-- [![TensorFlow Hub](https://img.shields.io/badge/TF%20Hub-Models-FF6F00?logo=tensorflow)](https://tfhub.dev/google/collections/movinet) -->

[![Colab Pro](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1drAzT7ZtNRWvDDxkHFLaz2VSFr56EHJu#scrollTo=XCmvsZY0KdhS)


YOLOv3: An Incremental Improvement [![YOLOv3](http://img.shields.io/badge/Paper-arXiv.1804.02767-B3181B?logo=arXiv)](https://arxiv.org/abs/1804.02767)

The TensorFlow Model Garden (TFMG) has a modular structure, supporting component re-use between exemplar implementations. Modularity both simplifies implementation, andaccelerates innovation: model components can be recombined into a new model per-forming a different function. For example, the YOLO family is targeted towards object detection, but can be used for image classification by connecting an image classification head to the current backbone.


In this Colab Notebook, we will be using the YOLOv3 model to show how to create a TensorFlow Model Garden project by following [TFMG components](https://github.com/tensorflow/models/tree/master/official/vision/beta/projects/example). Also, TensorFlow provides compenhensive framework and API to support your TFMG project, including [TensorFlow Hub](https://www.tensorflow.org/hub) and [TensorFlow Datasets](https://www.tensorflow.org/datasets). 

In this tutorial, we provide a step-by-step example which works for both building a model from scratch and directly loading the pre-trained model from TensorFlow Hub for detection.

# Step 0: Setup

It is recommended to run the models using GPUs or TPUs. 

To select a GPU/TPU in Colab, select `Runtime > Change runtime type > Hardware accelerator` dropdown in the top menu. 

If you upgraded to **[Colab PRO](https://colab.research.google.com/signup)**, check [![Colab Pro](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/notebooks/pro.ipynb) for using priority access to our fastest GPUs. 