# Customize Data Pipelines

## Extract-Transform-Load (ETL) Data Pipeline

In the TensorFlow Model Garden, the Extract-Transform-Load (ETL) architecture is the preferred approach for the Data Pipeline. This modular design promotes reusein different TensorFlow Model Garden exemplars.  Different models may share datasets (Extract), augmentations (Transform), or final stages (Load).

Each exemplar’s decoder and parser should be inherited from the `decoder` and `parser` classes of the TensorFlow Model Garden. The loader serves as an interface between thedata pipeline and the model, and will vary based on architecture.

### ETL - Extract 

The extraction component, called the decoder, converts the raw dataset to a format that is compatible the rest of the Data Pipeline. It is imperative before developing the decoder to determine the raw data format structure. After this, the data should serialized into a format that will be compatible with the data handling functions. 

TensorFlow’s data handling functions are found in `tensorflow.data`. This API interfaces with and is optimized for TensorFlow’s own file format, known as `TFRecord`, which is a serialized sequence of binary records. The `tensorflow.train` API and `tensorflow.io` API convert raw inputs to `TFRecord`, which can later be loaded in as a `tensorflow.dataset` object. After converting and standardizing the raw input into a `TFRecord`, it is ready to be passed onto the transformation component. 
Note that there should be one decoder class for a specific dataset and this should be inherited from the Model Garden’s `decoder` class. This promotes two kinds of standardization: among decoders within the project, and among inputs going into the transformation component of the data pipeline. 


### ETL - Transform 

The transformation component is called the parser and it handles preprocessing, normalization and formatting of the input features and labels within the dataset to fit the model input format. The parser should be a class that contains two data handling methods: one for training and another for evaluation. There should also be another method that returns one of these methods based on the activity being done. The returned method should contain all of the used data handling functions to prevent incompatible library dependencies and optimize performance. 

The function returned by the parser can be mapped onto the dataset, but only modifies the features and labels that are loaded from a persistent storage source into system memory and not the dataset itself. This will allow the model to begin training faster as only the first batch will be required to be processed and using prefetching, data can be processed on the CPU while the model trains on the GPU, which will reduces both processing units’ idle time. Another way to optimize the efficiency of the transformation component is by parallelizing the mapped functions which reduces the amount of time it will take the CPU to execute the mapped function over the batch. Note that there should be a parser depending on the type of training and phases of that training. Similar to the decoder, there should be one parser class for a specific dataset and this should be abstracted from the Model Garden’s `decoder` class. 


### ETL - Load 

The loading component, which is found in the task structure within the method called build_inputs, is responsible for loading of the dataset into the model for the purpose of training. This component leverages dataset augmentations such as batching, prefetching, caching, shuffling and interleaving. The `build_inputs` method instantiates the decoder, parser, as well as the entire loading component, `input_reader`. The dataset augmentations are primarily determined by the size of the dataset such as how the dataset interacts with system memory. 

If a dataset fits in the available system memory then it can be cached. Otherwise, the dataset must be processed in batches. If the dataset is larger than system memory but caching is attempted, then the machine will crash. Shuffling and interleaving should be done with the understanding that the larger the dataset, the longer the time it takes to start training for each epoch. 
