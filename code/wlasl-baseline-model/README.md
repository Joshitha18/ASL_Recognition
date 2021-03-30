# Automatic Captioning of American Sign Language (ASL) Videos

## Introduction

In this project, we seek to use deep neural networks to automatically caption ASL videos.  Specifically, the aim is to examine frames of videos
to first classify each [gloss](https://www.slideshare.net/MsAmyLC/glossing-in-asl-what-is-it-eight-examples) shown in the video, then stringing
the glosses together to determine the most probable sentence conveyed by the signer.  Glosses can be considered as "shortcuts" in order to
convey the information quickly and it's up to whoever is receiving the glosses to understand the final message that is conveyed.

The first step is to train a deep neural network to recognize what gloss is being conveyed in a sequence of frames.  Once we understand what gloss
is being conveyed, we then examine a video sequence to extract out the sequence of glosses which we then translate into an actual English
caption.  The intent is to use a `seq2seq` model to convey this information.

## Required Modules

You can install the required modules by accessing the `requirements.txt` file and using `pip`:

```
pip install -r requirements.txt
```

However if you want to install the modules yourself, you will need:

* `numpy`
* `torch`
* `torchvision`
* `opencv-contrib-python`
* `matplotlib`

## Framework Summary

We are extending the work done by Dongxu Li in their work at WACV 2020 titled ["Word-Level Deep Sign Language Recognition from Video: A New Large-Scale Dataset
and Methods Comparison"](https://dxli94.github.io/WLASL/).  In their work, they modified the [Inception3D](https://github.com/deepmind/kinetics-i3d) architecture
that was defined to classify human actions.  As a natural extension to this, the work could be used to classify ASL glosses as each gloss could be considered a
human action with facial expressions and hand actions being performed.  This repo is a copy of Dongxu Li's training tools, fixing some runtime bugs as well as providing the
weights to the network once training is complete.  The original source code for training their network [can be found here](https://drive.google.com/file/d/1vktQxvRHNS9psOQVKx5-dsERlmiYFRXC/view).
The code does not contain the final weights for gloss recognition, so we took it upon ourselves to train the network.  We have trained the 2000 gloss model, where
the 2000 most frequently used glosses found in the dataset. Please note that the trained model is only CNN based.  The paper mentioned that the source code also introduces
a CNN-RNN hybrid which was not available in the original source code for training their models, even though the author proported to make that available.

## Setup - Running the trained model

1. Please clone the [toolkit for downloading the dataset curated by Dongxu Li](https://github.com/dxli94/WLASL) and follow the instructions for downloading
the videos which embody the largest video dataset for Word-Level ASL recognition.  Please note that downloading the entire video dataset will take a considerable
amount of time (at least one full day) so patience is recommended.

2. Once you download the videos, clone this repo locally then place the videos inside the `videos` directory and ensure all of the videos are at their top most
level in the directory. That is, copy the videos over ensuring they are all there with no further subdirectories. The directory should look contain files
like `videos/00295.mp4`, `videos/00333.mp4`, etc.

3. Access the `test.ipynb` notebook file which will load in the pretrained weights, set up the model and perform inference on the test dataset.  The decomposition
of what is the training dataset and test dataset is found in the JSON file in the `preprocess` directory: `nslt_2000.json`.

4.  The top-1, top-5 and top-10 accuracy for the test dataset is reported in the notebook which is in alignment with what is reported in Li's paper.

## Setup - Reproducing the trained network

All you have to do is examine the `train.ipynb` notebook and run the cells from start to finish which will train the network and produce the final PyTorch checkpoint.
Please note that the network was trained on a T4 TPU which took approximately two days to train.

## Summary of changes made to original source

* `nslt_dataset.py`: The video reading function has been made more robust so that if a video is invalid, we exit gracefully
* `nslt_dataset.py`: The `torch.utils.data.Dataset` that constructs the paths for the train and test videos is now saved and pickled so that this does not need to be
built again when performing the train task. This is very computationally intensive.
* There have been debug statements scattered throughout the source to provide more context and progress on where the training and validation of the test dataset is at.
* There were several PyTorch statements that have been deprecated due to the recent versions of PyTorch.  They have been replaced with their accepted equivalents.
* `train_i3d.py`: `argparser` has been removed in favour of running the training process in the `train.ipynb` notebook.
* `test_i3d.py`: `argparser` has been removed in favour of running the validation process in the `test.ipynb` notebook.

## `seq2seq` model for gloss sequence to English caption

This is currently a work in progress.  We hope to have this complete in the near future.
