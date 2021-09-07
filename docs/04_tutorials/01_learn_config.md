# Learn about Configs



In order to configure a task, each implementation will require the addition of a set of dat-aclass configurations that inherit from those found in `offical.core.configdefinitions`.

In addition to the configuration files, you should include a method named `experiment` that predefines all the configuration parameters and serves as a default model state if agiven parameter is not in the operating configuration file. This will allow configurationfiles to remain concise while also preserving the model’s essential functionality.

The configurations and the Tasks are designed to come together into an operational model that can be manually configured via the input of a model configuration, or automatically configured using the trainer and a configuration file. Given that the model operatesusing this dual functionality, the task will not have any class parameters other than `self.taskconfig` used to hold the input configuration dataclasses.



More details in TensorFlow offical link: [TensorFlow Model - config definitions](https://github.com/tensorflow/models/blob/master/official/core/config_definitions.py)


## flags command line args


flags:

[require]
- 'experiment':
    - the experiment type registered, specifying an ExperimentCongifg
- 'mode': A 'str', specifying the mode. Can be 'train', 'eval', 'train_and_eval'
      or 'continuous_eval'.
    - 'train',
    - 'eval',
    - 'train_and_eval',
    - 'continuous_eval',
    - 'continuous_train_and_eval',
    - 'train_and_validate'.
- 'model_dir': A 'str', a path to store model checkpoints and summaries.

[optional]
- 'config_file': ./congigs/experiments

train_lib.run_experiment(
    distribution_strategy=distribution_strategy,
    task=task,
    mode=FLAGS.mode,
    params=params,
    model_dir=model_dir)

trainer = train_utils.create_trainer(
    params,
    task,
    train='train' in mode,
    evaluate=('eval' in mode) or run_post_eval,
    checkpoint_exporter=maybe_create_best_ckpt_exporter(
        params, model_dir))

@gin.configurable
def create_trainer(params: config_definitions.ExperimentConfig,
                   task: base_task.Task,
                   train: bool,
                   evaluate: bool,
                   checkpoint_exporter: Optional[BestCheckpointExporter] = None,
                   trainer_cls=base_trainer.Trainer) -> base_trainer.Trainer:
  """Create trainer."""
  logging.info('Running default trainer.')
  model = task.build_model()
  optimizer = task.create_optimizer(params.trainer.optimizer_config,
                                    params.runtime)
  return trainer_cls(
      params,
      task,
      model=model,
      optimizer=optimizer,
      train=train,
      evaluate=evaluate,
      checkpoint_exporter=checkpoint_exporter)
```
from absl import app

# pylint: disable=unused-import
from official.common import flags as tfm_flags


if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(train.main)
```
