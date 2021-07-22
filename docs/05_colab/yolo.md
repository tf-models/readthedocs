# Object Detection: YOLO 

## Description

YOLO algorithem divides any given input image into $S \times S$ grid system. Each grind on the input image is respoinsible for detection on object. The grid cell predicts the number of boundary boxes for an object. 

For every boundary box has fiver elements: 
- `x, y`: the coordinates of the object in the input image;
- `w, h`: the width and height of the object;
- `score`: confidence score, the probability that box contains an object and how accurate. 


## YOLO

- [YOLO: You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

### YOLO Architecture 



### Limitation 

YOLO v1 could not find small objects if they are appeared as a cluster. 


## YOLO v2 

- [YOLOv2 / YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242v1)


Changes from YOLO to YOLO v2 (or called YOLO9000): 
- **Batch Normalization**:
- **Higher Resolution Classifier**:
- **Anchor Boxes**:
- **Fine-Grained Features**:
- **Multi-Scale Training**:
- **Darknet-19**:

### YOLO v2 Architecture (DarkNet-19)



## YOLO v3 

- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)

The previous version has been improved for an incremental improvement which is now called YOLO v3. As many object detection algorithms are been there for a while now the competition is all about how accurate and quickly objects are detected. YOLO v3 has all we need for object detection in real-time with accurately and classifying the objects. The authors named this as an incremental improvement.

- **Bounding Box Predictions**: In YOLO v3 gives the score for the objects for each bounding boxes. It uses logistic regression to predict the objectiveness score.
- **Class Predictions**: In YOLO v3 it uses logistic classifiers for every class instead of softmax which has been used in the previous YOLO v2. By doing so in YOLO v3 we can have multi-label classification. With softmax layer if the network is trained for both a person and man, it gives the probability between person and man let’s say 0.4 and 0.47. With the independent classifier gives the probability for each class of objects. For example if the network is trained for person and a man it would give the probability of 0.85 to person and 0.8 for the man and label the object in the picture as both man and person.
- **Feature Pyramid Networks (FPN)**: YOLO v3 makes predictions similar to the FPN where 3 predictions are made for every location the input image and features are extracted from each prediction. By doing so YOLO v3 has the better ability at different scales. As explained from the paper by each prediction is composed with boundary box, objectness and 80 class scores. Doing upsampling from previous layers allows getting meaning full semantic information and finer-grained information from earlier feature map. Now, adding few more convolutional layers to process improves the output 

### YOLO v3 Architecture (DarkNet-53)




## YOLO v4

- [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)