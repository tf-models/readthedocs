# TF-Vision Training Library

TensorFlow Model Garden is conducting a major rewrite to consolidate model implementations with multi-level abstractions and coherent coding style. In order to manage the diverse models and machine learning tasks, we introduce a generic workflow and some common libraries.

## Code WorkFlow

### Task

We define all modeling artifacts of a particular machine learning task as a `Task` object. The `Task` includes `build_model` for creating the model instance, `build_inputs` for defining `tf.data` input pipeline, `train_step` and `validation_step` for the computation, `build_metrics` for streaming metrics etc.

Users usually need to override the following key functions:
- `__init__`: [don't-usually-override]
- `build_inputs`: [often-do-override]
- `build_model`
- `train_step`
- `validation_step`
- `build_metrics`: [often-do-override]
- `initialize`

In addition, if users want to compute the evaluation metrics on host (using numpy or TF metrics that cannot run on TPUs), users can override (1) `aggregate_logs` method that updates a stateful memory with the output of `validation_step`, and (2) `reduce_aggregated_logs` method that processes the aggregated stateful memory to get final results. Please see `sentence_prediction.py` as an example.

For example, we can find all NLP tasks inside `nlp/tasks`. Each task is registered by the task registry as:

```python
@task_factory.register_task_cls(MaskedLMConfig)
class MaskedLMTask(base_task.Task):
```

The `task registry` essential holds a dictionary to map the `Task` config class to the `Task` class.

Default Behavior: initialize from checkpoints

Initializing from a pre-trained checkpoint is a very common use case and the usages can be very flexible. We define a `initialize` function in the base `Task` and the subclasses can override it.

```python
def initialize(self, model: tf.keras.Model):
  # Calls tf.train.Checkpoint APIs to restore states.
```

The `TaskConfig` contains a string attribute `init_checkpoint`. The initialize function restore the states from the path or directory provided by `init_checkpoint`.

The default workflow in the `run_experiment` is: 
- `task.initialize` --> `trainer.initialize` --> `tf.train.CheckpointManager(init_fn=xxx)`.
  
The `tf.train.CheckpointManager` will only trigger the initialization function if there is no checkpoint state file in the checkpoint directly. Thus, if you restart in the middle of the training, you do not expect the initialization function to be called.

*Note that, the key API to initialize the model variables from a checkpoint is the `init_fn` of `tf.train.CheckpointManager`. Users can branch the model garden code to customize for their use cases. Please feel free to directly implement a initialization closure to pass to the `tf.train.CheckpointManager`.*

### Training Binary

Once a task is registered, it can be used inside the training binary: `train binary`. The task is used in the entire training binary as follows:

```python
params = ExperimentConfig()
with strategy.scope():
  task = task_factory.get_task(params.task, logging_dir=model_dir)
  trainer = Trainer(task, ...)

checkpoint_manager = tf.train.CheckpointManager(...)
controller = orbit.Controller(trainer, checkpoint_manager,...)
controller.train(steps=num_steps)
```

We create trainer instances based on [Orbit] and it consumes the training artifacts defined in the task instance. The trainer instance will be further passed to [Orbit controller] and finally the controller will execute the training and evaluation procedure.

You can find predefined trainers in [trainers] folder, there is a collection of trainers implemented, e.g. `the default trainer` for most supervised learning tasks, `progressive trainer` to speed up training, etc. Users can build their customized trainers as well.

We encapsulate the workflow of creating the task, trainer, controller and running the controller in different modes as the `run_experiment` function. The function is supposed to be used as a reference and users are free to copy. We do not expect the `run_experiment` to support diverse use cases. Inside the model garden, we have defined variants of `run_experiment` for different use cases.

### Experiment Configuration

We use `Config`, a dataclass with serialization functionality, as the container to restore standard coarse configuration. The task and trainer behavior is configured by the `ExperimentConfig`, which includes the configurations for `task`, `trainer` and `runtime`.

#### Caveats

We use `dataclasses` to define configuration python classes to get nice properties of dataclasses. However, `pytype` does not work well for the inheritance of dataclasses. The attributes from parent classes cannot be recognized during pytype inference.

`Config` supports convenient transformation from nested dictionary inputs to the type annotated class. You may see this pattern in the model garden experiment examples. However, this violates type annotation because the dataclass init expect `Config` objects instead of the plain dictionary.

*Note: if you see `wrong-keyword-args` pytype error in the experiment configuration file. Please disable pytype either through comments or BUILD.*

#### Task configuration

`task` configuration is passed to the each `Task constructor` to create models and build tf.Datasets. For example, code

```python
@dataclasses.dataclass
class SentencePredictionConfig(cfg.TaskConfig):
  init_checkpoint: str = ''
  init_cls_pooler: bool = False
  hub_module_url: str = ''
  metric_type: str = 'accuracy'
  model: ModelConfig = ModelConfig()
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()
```

#### Trainer configuration

`trainer` configuration is used inside `train_lib.py` and passed to Trainer instance.

#### Runtime configuration

`runtime` configuration is used inside `train_lib.py` and configures `tf.distribute`, mixed precision, etc.

### Experiment Registry

We use an `experiment registry` to build a mapping between experiment type to experiment configuration instance. Users usually will define a concrete experiment through a getter function as:

```python
from official.core import config_definitions as cfg
from official.core import exp_factory

@exp_factory.register_config_factory("bert/pretraining")
def bert_pretraining() -> cfg.ExperimentConfig:
  config = cfg.ExperimentConfig(
      task=masked_lm.MaskedLMConfig(
          train_data=pretrain_dataloader.BertPretrainDataConfig(),
          validation_data=pretrain_dataloader.BertPretrainDataConfig(
              is_training=False)),
      trainer=cfg.TrainerConfig(
          optimizer_config=BertOptimizationConfig(), train_steps=1000000))
  return config
```

### Overriding Configuration via Yaml and Gin

The experiment configuration object will be retrieved at the begining of the `train binary`. It could be further override by yaml files provided by `--config_file`.

For example, to train a English BERT on Wiki-Books dataset, we can set the model, data and training details by:

```shell
--config_file=/nlp/bert/experiments/wiki_books_pretrain.yaml
```

Note that we have many modules are `gin.configurable`, so users are able to provide gin configuration to override detailed modeling hyperparameters, provided by `--gin_file`.

For example, to further customized the Transformer encoder architecture with variants on attention and feedforward blocks, we can add:

```shell
--gin_file=/nlp/gin/talking_gated_encoder.gin
```

## Run locally

For fast, local debugging of training loop, you can define some trivial testing experiments small enough to run on local cpus:

```shell
blaze run -c opt \
/official/nlp/train \
 -- --experiment=bert/local --model_dir=/tmp/tfnlp_sandbox/ \
 --mode=train --alsologtostderr 2>&1
```