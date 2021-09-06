# Customize Models

First, read the TensorFlow and Keras documents:

    - [Keras Developer Guides](https://keras.io/guides/)
    - [Making new Layers and Models via subclassing](https://www.tensorflow.org/guide/keras/custom_layers_and_models)

In model garden, we recommend to use `register_keras_serializable()` to implement customize layers and model as below:

```python

@tf.keras.utils.register_keras_serializable(package='Vision')
class MyLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        # this _config_dict is required, 
        # as the `get_config()` will be used 
        # in `tf.keras.utils.register_keras_serializable()`
        self._config_dict = {
            'filters': filters,
            'kernel_size': kernel_size,
            'strides': strides
        }

    # required for `tf.keras.utils.register_keras_serializable()`
    def get_config(self):
        return self._config_dict
```

Here, the decorater `tf.keras.utils.register_keras_serializable()` injects the Keras custom object dictionary `MyLayer()`, so that it can be serialized and deserialized without needing an entry in the user-provided custom object dict. 

Note that to be serialized and deserialized, classes must implement the `get_config()` method. Functions do not have this requirement.
The object will be registered under the key 'package>name' where `name`, defaults to the object name if not passed.
##


The model is responsible for accomplishing the implementation’s specific subtask, such as object detection for YOLO. It is a trainable structure of layers, and layers can be grouped into blocks based on their frequency of occurrence. Each layer and block can be designed and tested individually, and then integrated into one structure prior to training. These model sub-elements are shown in Figure 4 and explained in Appendix A. The layer sizes must be noted to maintain the shape of the model and ensure that adjustments are made so that the output of one element is the same size as the input of the next. A model, unlike a building block, is totally self-contained and usually much fewer parameters to modify. It is much more rigid because it is not meant to be used to bigger trainable structures. 

The architecture of a machine learning model will vary based on its purpose. For computer vision, models have three main components: feature extractor (backbone), decoder and head. A model decoder refers to the post-processing component of the model, and is different from the data pipeline decoder. The backbone refers to the combination of layers responsible for feature extraction. The backbone is usually a large number of sequential layers and the majority of the model training process is to adjust the backbone parameters. The decoder, which usually has less layers than the backbone, handles post-processing and serves as a connection between the backbone and the head. The head, often the smallest element, is responsible for the final model task (ex: image classification). Depending on the model architecture, the components include within the backbone and decoder may vary. 

Throughout the model development process, continually refer to the model architecture provided within the research paper. Usually, the architecture will mention the number of layers, size of each layer, how each layer is connected to other layers and the optimizer such for parameter adjustment. Identifying repeating layers and layer connections within the architecture can help create a “building block” to reduce the lines of code of the model itself Many model architectures will require at least one custom layer that does not exist in the TensorFlow API. 

Many models will just be a wrapper class around the two sub-model components that brings the two pieces together. While constructing the model, ensure that the structure mentioned in Figure 4 is followed. Having such a structure will maintain code modularity and allow your code to be usable for other developers within the TFMG community. 
