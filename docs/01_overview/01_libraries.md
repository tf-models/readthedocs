# Modeling Libraries

## TF-NLP for Natural Language Processing

TF-NLP is a modeling library for natural language processing using best practices in TensorFLow 2. We. It provides a collection of state-of-the-art, reproducible, extensible, and scalable natural language models, as well as reusable modeling components and libraries.

## TF-Vision for Computer Vision

TF-Vision is a modeling library for computer vision natively designed in Tensorflow 2. It provides a wide range of state-of-the-art computer vision models, flexible design for easy extension, and end-to-end pipeline to accomplish tasks in computer vision benchmarks.


### official.modeling.tf_utils

[source code](https://github.com/tensorflow/models/blob/master/official/modeling/tf_utils.py)

```python
tf_utils.pack_inputs(
    inputs
)

tf_utils.is_special_none_tensor(
    tensor
)

tf_utils.get_activation(
    identifier, use_keras_layer=False
)

tf_utils.get_shape_list(
    tensor, expected_rank=None, name=None
)

tf_utils.assert_rank(
    tensor, expected_rank, name=None
)

tf_utils.safe_mean(
    losses
)
```