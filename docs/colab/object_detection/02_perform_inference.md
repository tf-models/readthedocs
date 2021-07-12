# Perform Inference

TensorFlow provides datasets here [https://www.tensorflow.org/datasets](https://www.tensorflow.org/datasets). All dataset builders are subclass of `tfds.core.DatasetBuilder`. To get the list of available builders, use `tfds.list_builders()` or look at our [catalog](https://www.tensorflow.org/datasets/catalog/overview).

## Example

For this tutorial, we will use the [rock-paper-scissors dataset](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors) in TFDS (Catalog: Image Classification). This cell will download the dataset and demo some of the images.

```python
# Download rock_paper_scissors dataset from TFDS. 
sample = tfds.load('rock_paper_scissors', split='train').batch(8)

# Visualize the original dataset
key_to_label = {0: 'rock',
                1: 'paper',
                2: 'scissors'}
for d in sample.take(1):
    fig, ax = plt.subplots(2, 4)
    image = d['image']
    label = d['label']
    np_label = label.numpy()
    for j, im in enumerate(image):
        ax[j // 4, j % 4].set_title(key_to_label[np_label[j]], fontsize = 20)
        ax[j // 4, j % 4].imshow(im)
    fig.set_size_inches(15, 7, forward=True)
    fig.tight_layout()
    plt.show()
```