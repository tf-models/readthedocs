# Prerequisites

- Python 3.6-3.8
- Ubuntu 16.04 or later
- Windows 7 or later (with C++ redistributable)
- macOS 10.12.6 (Sierra) or latter (no GPU support)
- TensorFlow 2

# Installation

* The models in the master branch are developed using TensorFlow 2,
and they target the TensorFlow [nightly binaries](https://github.com/tensorflow/tensorflow#installation)
built from the
[master branch of TensorFlow](https://github.com/tensorflow/tensorflow/tree/master).
* The stable versions targeting releases of TensorFlow are available
as tagged branches or [downloadable releases](https://github.com/tensorflow/models/releases).
* Model repository version numbers match the target TensorFlow release,
such that
[release v2.2.0](https://github.com/tensorflow/models/releases/tag/v2.2.0)
are compatible with
[TensorFlow v2.2.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0).

Please follow the below steps before running models in this repository.

## Requirements

* The latest TensorFlow Model Garden release and TensorFlow 2
  * If you are on a version of TensorFlow earlier than 2.2, please
upgrade your TensorFlow to [the latest TensorFlow 2](https://www.tensorflow.org/install/).

```shell
pip3 install tf-nightly
```

## Installation

### Method 1 (recommend): Install the TensorFlow Model Garden pip package

The **tf-models-official** is the stable Model Garden package. `pip` will install all models and dependencies automatically. 

Note that **tf-models-official** may not include the latest changes in this github repo. To include latest changes, you may install **tf-models-nightly**, which is the nightly Model Garden package created daily automatically.

```shell
pip install tf-models-official 
pip install tf-models-nightly 
```

(Optional) If you are using **nlp** packages, please also install **tensorflow-text**:

```shell
pip install tensorflow-text
```

Please check out our [example](colab/fine_tuning_bert.ipynb) to learn how to use a PIP package.

### Method 2: Clone the source

Clone the GitHub repository:

```shell
git clone https://github.com/tensorflow/models.git
```

Add the top-level ***/models*** folder to the Python path.

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/models
```

(Optional) If you are using a Colab notebook, please set the Python path with os.environ.

```python
import os
os.environ['PYTHONPATH'] += ":/path/to/models"
```

Install other dependencies

```shell
pip3 install --user -r official/requirements.txt
```

(Optional) if you are using **nlp** packages, please also install **tensorflow-text-nightly**:

```shell
pip3 install tensorflow-text-nightly
```


## Contributions

If you want to contribute, please review the [contribution guidelines](https://github.com/tensorflow/models/wiki/How-to-contribute).