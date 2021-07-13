
Welcome to TensorFlow Model Garden's documentation!
================================


The TensorFlow Model Garden (TFMG) is a collection of models that use TensorFlow’s high-level APIs. 
They are intended to be well-maintained, tested, and kept up to date with the latest TensorFlow API.

The goal of the TFMG is to develop exemplary implementations of prominent machine learning models in community. 
They should also be reasonably optimized for fast performance while still being easy to read. 

These models are used as end-to-end tests, ensuring that the models run with the same or improved speed and performance with each new TensorFlow build.

This documentation is to explain a process for reproducing a state-of-the-art machine learning model at a level of quality suitable for inclusion in the TFMG, 
which includes the engineering process and elaborate on each step, from paper analysis to model release. 


.. toctree::
   :maxdepth: 2
   :caption: Overview

   intro/modeling_steps.md

.. toctree::
   :maxdepth: 2
   :caption: Quick Start

   quick_start/01_install.md
   quick_start/02_TF_TPU.md   

.. toctree::
   :maxdepth: 2
   :caption: Computer Vision

   colab_cv/0object_detection/01_install_tf_vision.md
   colab_cv/0object_detection/02_perform_inference.md   
   colab_cv/0object_detection/03_train_detector_on_dataset.md
   colab_cv/0object_detection/04_test_trained_detector.md

   colab_cv/image_classification/01_install_tf_vision.md
   colab_cv/image_classification/02_perform_inference.md   
   colab_cv/image_classification/03_train_detector_on_dataset.md
   colab_cv/image_classification/04_test_trained_detector.md


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   guides/01_learn_config.md
   guides/02_datasets.md
   guides/03_data_pipelines.md
   guides/04_models.md
   guides/05_runtime_settings.md
   guides/06_losses.md
   guides/07_finetuning_models.md


.. toctree::
   :maxdepth: 2
   :caption: Contribution

   contribution_license/contribution_guide.md

.. toctree::
   :maxdepth: 2
   :caption: Test Example

   example