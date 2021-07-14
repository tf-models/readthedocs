# Learn about Configs

In order to configure a task, each implementation will require the addition of a set of dat-aclass configurations that inherit from those found in `offical.core.configdefinitions`. 

In addition to the configuration files, you should include a method named `experiment` that predefines all the configuration parameters and serves as a default model state if agiven parameter is not in the operating configuration file. This will allow configurationfiles to remain concise while also preserving the model’s essential functionality. 

The configurations and the Tasks are designed to come together into an operational model that can be manually configured via the input of a model configuration, or automatically configured using the trainer and a configuration file. Given that the model operatesusing this dual functionality, the task will not have any class parameters other than `self.taskconfig` used to hold the input configuration dataclasses.



More details in TensorFlow offical link: [TensorFlow Model - config definitions](https://github.com/tensorflow/models/blob/master/official/core/config_definitions.py)