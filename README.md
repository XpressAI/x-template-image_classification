# Xircuits Image Classification Project Template

## Template Setup
You will need python 3.9+ to install xircuits. We recommend installing in a virtual environment.
### Libraries setup
To install the required libraries Run: 
```
$ pip install -r requirements.txt
```
### Launch
To launch Xircuits Run:
```
$ xircuits
```
More detailed information on installation, setup and features can be found on [Xircuits](https://github.com/XpressAI/xircuits)  

## Image Classification

In this template, you will able to classify images of different objects by using transfer learning from a pre-trained network.

We will leverage the pre-trained model in two ways to train our custom classification model:

1. Feature Extraction: Use the representations learned by a previous pre-trained model to extract meaningful features from new samples. We add a new classifier head, which will be trained from scratch, on top of the pre-trained model so that we could repurpose the feature maps learned previously for the dataset.

2.  Fine-Tuning: we don't need to (re)train the entire model as the base convolutional network already contains features that are generically useful for classifying pictures. However, the final, classification part of the pre-trained model is specific to the original classification task, and subsequently specific to the set of classes on which the model was trained. **Unfreeze** a few of the top layers of a frozen model base and jointly train both the newly-added *classifier head* layers and the last layers of the base model. This allows us to "fine-tune" the higher-order feature representations in the base model in order to make them more relevant for the specific task.

This template follows the *image classifier* training workflow.

- Examine and understand the data
- Build an input pipeline
- Compose the model
- Load in the pre-trained base model (and pre-trained weights)
- Stack the classification layers on top
- Train the model
- Evaluate model
- Save model

## object_classification_template.xircuits

- In this template we download the *cats_and_dogs_filtered* dataset from *Tensorflow* and perform a simple binary image classification model training and fine-tuning.  

![Template](images/template_image_class.gif)


### Notice:

If you would like to use your own dataset, it should follow this structure: 

```
<Dataset_folder>
    |_train
        |_Class-1
            |_image-1_class-1
            |_image-2_class-1
            |_...
        |_Class-2
            |_image-1_class-2
            |_...
        |_class-3
        |...
    |_validation
        |_Class-1
            |_image-1_class-1
            |_image-2_class-1
            |_...
        |_Class-2
            |_image-1_class-2
            |_...
        |_class-3
        |...
```