# Modeling Libraries


### official.modeling.tf_utils

[source code](https://github.com/tensorflow/models/blob/master/official/modeling/tf_utils.py)

```python
# could be used in modeling layers
tf_utils.get_activation(identifier, use_keras_layer=False)

# could useful in modeling layers
tf_utils.get_shape_list(tensor, expected_rank=None, name=None)

# could be used in tasks 
tf_utils.safe_mean(losses)
```

```python
# helper
tf_utils.assert_rank(tensor, expected_rank, name=None)
tf_utils.is_special_none_tensor(tensor)
```
### official.modeling.performance

```python

# could be used in tasks
performance.configure_optimizer(
    optimizer, use_float16=False, use_graph_rewrite=False, loss_scale=None
)

# could be used in tasks
set_mixed_precision_policy(
    dtype, loss_scale=None
)

```

### official.modeling.grad_utils

```python
minimize_using_explicit_allreduce(
    tape, optimizer, loss, trainable_variables, 
    pre_allreduce_callbacks=None, post_allreduce_callbacks=None, 
    allreduce_bytes_per_pack=0
)
```

## TF-NLP for Natural Language Processing

TF-NLP is a modeling library for natural language processing using best practices in TensorFLow 2. We. It provides a collection of state-of-the-art, reproducible, extensible, and scalable natural language models, as well as reusable modeling components and libraries.

### modeling.tf_utils

```python
# pack_inputs / unpack_inputs can be use in NLP decoder
tf_utils.pack_inputs(inputs)
tf_utils.unpack_inputs(inputs)
```

## TF-Vision for Computer Vision

TF-Vision is a modeling library for computer vision natively designed in Tensorflow 2. It provides a wide range of state-of-the-art computer vision models, flexible design for easy extension, and end-to-end pipeline to accomplish tasks in computer vision benchmarks.

