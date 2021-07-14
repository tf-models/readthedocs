
Welcome to MG's documentation!
================================


The [xxx] is a collection of models that use [xxx]’s high-level APIs. 
They are intended to be well-maintained, tested, and kept up to date with the latest [xxx] API.

The goal of the TFMG is to develop exemplary implementations of prominent machine learning models in community. 
They should also be reasonably optimized for fast performance while still being easy to read. 

These models are used as end-to-end tests, ensuring that the models run with the same or improved speed and performance with each new TensorFlow build.

This documentation is to explain a process for reproducing a state-of-the-art machine learning model at a level of quality suitable for inclusion in the TFMG, 
which includes the engineering process and elaborate on each step, from paper analysis to model release. 


.. toctree::
   :maxdepth: 2
   :caption: Overview

   01_overview/01_tfmg_lib.md
   01_overview/02_tf_datasets.md


.. toctree::
   :maxdepth: 2
   :caption: Quick Start

   02_quick_start/01_install.md
   02_quick_start/02_TF_TPU.md 
   02_quick_start/03_simple_example.md  


.. toctree::
   :maxdepth: 2
   :caption: Model Zoo 

   03_model_zoo/01_cv.md
   03_model_zoo/02_nlp.md


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   04_tutorials/01_learn_config.md
   04_tutorials/02_datasets.md
   04_tutorials/03_data_pipelines.md
   04_tutorials/04_models.md
   04_tutorials/05_runtime_settings.md
   04_tutorials/06_losses.md
   04_tutorials/07_finetuning_models.md


.. toctree::
   :maxdepth: 2
   :caption: Colab Example

   05_colab/01_general_example.md
   05_colab/02_object_detection.md
   05_colab/03_image_classification.md


.. toctree::
   :maxdepth: 2
   :caption: Colab Example

   06_FAQ/faq.md


.. toctree::
   :maxdepth: 2
   :caption: Contribution

   07_notes/contribution_guide.md

