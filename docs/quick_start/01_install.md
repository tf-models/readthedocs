# Prerequisites

## How to get started with the official models

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

### Requirements

* The latest TensorFlow Model Garden release and TensorFlow 2
  * If you are on a version of TensorFlow earlier than 2.2, please
upgrade your TensorFlow to [the latest TensorFlow 2](https://www.tensorflow.org/install/).

```shell
pip3 install tf-nightly
```

### Installation

#### Method 1: Install the TensorFlow Model Garden pip package

**tf-models-official** is the stable Model Garden package.
pip will install all models and dependencies automatically.

```shell
pip install tf-models-official
```

If you are using nlp packages, please also install **tensorflow-text**:

```shell
pip install tensorflow-text
```

Please check out our [example](colab/fine_tuning_bert.ipynb)
to learn how to use a PIP package.

Note that **tf-models-official** may not include the latest changes in this
github repo. To include latest changes, you may install **tf-models-nightly**,
which is the nightly Model Garden package created daily automatically.

```shell
pip install tf-models-nightly
```

#### Method 2: Clone the source

1. Clone the GitHub repository:

```shell
git clone https://github.com/tensorflow/models.git
```

2. Add the top-level ***/models*** folder to the Python path.

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/models
```

If you are using a Colab notebook, please set the Python path with os.environ.

```python
import os
os.environ['PYTHONPATH'] += ":/path/to/models"
```

3. Install other dependencies

```shell
pip3 install --user -r official/requirements.txt
```

Finally, if you are using nlp packages, please also install
**tensorflow-text-nightly**:

```shell
pip3 install tensorflow-text-nightly
```

## Contributions

If you want to contribute, please review the [contribution guidelines](https://github.com/tensorflow/models/wiki/How-to-contribute).