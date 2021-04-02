# HelloFlowers

A hello world abridged implementation of the TensorFlow [Image Classification tutorial](https://www.tensorflow.org/tutorials/images/classification).

There is also a 96% accuracy model trained via GCP AutoML and hosted on GCP Cloud Storage. The forward pass Tensorflow.js implementation is hosted on [GitHub Pages](https://mrderive.github.io).

## Installation

Download the [flower pictures](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) from Google.

Extract the flower pictures:
```
tar -xvf flower_photos.tgz
```

Clone or download `flower_weights.data-00000-of-00001` and `flower_weights.index`.

Clone or download `helloflowers.py`.

Note: Should go without saying, but just in case: you need to have [TensorFlow 2 installed](https://www.tensorflow.org/install).

## Usage

Just run the Python script:
```
$ python3 helloflowers.py

Found 3670 files belonging to 5 classes.
Using 2936 files for training.
Found 3670 files belonging to 5 classes.
Using 734 files for validation.
23/23 [==============================] - 63s 3s/step - loss: 1.3647 - accuracy: 0.8610
```
