# Modified from exmaple_config.py 

"""Example experiment configuration definition."""
import tensorflow as tf
from typing import ClassVar, Dict, List, Optional, Tuple, Union
import dataclasses
import os
import numpy as np
import regex as re

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization

from official.vision.beta.configs import common

# default param classes
# pylint: disable=missing-class-docstring
@dataclasses.dataclass
class TfExampleDecoder(hyperparams.Config):
  regenerate_source_id: bool = False

@dataclasses.dataclass
class TfExampleDecoderLabelMap(hyperparams.Config):
  regenerate_source_id: bool = False
  label_map: str = ''

@dataclasses.dataclass
class DataDecoder(hyperparams.OneOfConfig):
  type: Optional[str] = 'simple_decoder'
  simple_decoder: TfExampleDecoder = TfExampleDecoder()
  label_map_decoder: TfExampleDecoderLabelMap = TfExampleDecoderLabelMap()

@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
    """Input config for training. Add more fields as needed."""
    input_path: str = '' #'gs://tensorflow2/coco_records/train/2017*'
    global_batch_size: int = 64 
    tfds_name: str = None  #'coco'
    tfds_split: str = None  #'train'
    is_training: bool = True
    dtype: str = 'float32'
    decoder: DataDecoder = DataDecoder()
    shuffle_buffer_size: int = 10000
    cycle_length: int = 10
    file_type: str = 'tfrecord'




@dataclasses.dataclass
class ExampleModel(hyperparams.Config):
  """The model config. Used by build_example_model function."""
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Losses(hyperparams.Config):
  l2_weight_decay: float = 0.0


@dataclasses.dataclass
class Evaluation(hyperparams.Config):
  top_k: int = 5


@dataclasses.dataclass
class ExampleTask(cfg.TaskConfig):
  """The task config."""
  model: ExampleModel = ExampleModel()
  train_data: ExampleDataConfig = ExampleDataConfig(is_training=True)
  validation_data: ExampleDataConfig = ExampleDataConfig(is_training=False)
  losses: Losses = Losses()
  evaluation: Evaluation = Evaluation()


@exp_factory.register_config_factory('tf_vision_example_experiment')
def tf_vision_example_experiment() -> cfg.ExperimentConfig:
  """Definition of a full example experiment."""
  train_batch_size = 256
  eval_batch_size = 256
  steps_per_epoch = 10
  config = cfg.ExperimentConfig(
      task=ExampleTask(
          model=ExampleModel(num_classes=10, input_size=[128, 128, 3]),
          losses=Losses(l2_weight_decay=1e-4),
          train_data=ExampleDataConfig(
              input_path='/path/to/train*',
              is_training=True,
              global_batch_size=train_batch_size),
          validation_data=ExampleDataConfig(
              input_path='/path/to/valid*',
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=90 * steps_per_epoch,
          validation_steps=steps_per_epoch,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 1.6,
                      'decay_steps': 350 * steps_per_epoch
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config