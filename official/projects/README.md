# TFMG Components


| Folders      | Required | Description                                                                                                                                                                                   |
|-------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **modeling** | yes      | Model and the building blocks.                                                                                                                                                                |
| **ops**     | yes      | Operations: utility functions used by the data pipeline, loss function and modeling.                                                                                                          |
| **losses**      | yes      | Loss function.                                                                                                                                                                                |
| **dataloaders** | yes      | Decoders and parsers for your data pipeline.                                                                                                                                                  |
| configs     | yes      | The  config  files for the task class to train and evaluate the model.                                                                                                                        |
| tasks       | yes      | Tasks for running the model. Tasks are essentially the main driver for training and evaluating the model.                                                                                     |
| common      | yes      | Registry imports. The tasks and configs need to be registered before execution.                                                                                                             |
| utils       | no       | Utility functions for external resources,  e.g. downloading weights, datasets from external sources, and the test cases for these functions. |
| demos       | no       | Files needed to create a Jupyter Notebook/Google Colab demo of the model. |

