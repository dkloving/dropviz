# Dropviz

## Description

[This paper](https://arxiv.org/abs/1506.08700) by Bouthillier, Konda, Vincent, and  Memisevic presents an interpretation of dropout as a form a data augmentation and provides a means to visualize the effective augmentation of dropout in a trained network. `Dropviz` allows the user to apply this visualization method to any model they wish.

## Installation

Dropviz can be installed easily with pip:

`pip install git+https://github.com/dkloving/dropviz`

## Usage

Dropviz uses the `eval()` and `train()` methods to access the intermediate outputs of your model in two modes.

 - User must chose an intermediate layer for which they want to view effective data augmentation of dropout.
 - Any dropout layers (or other layers that behave differently at train and eval times) are turned on during a forward pass during which the intermediate output, to be used as the target for the intermediate output in eval mode, is collected.  
 - Gradient descent is used to adjust the input image with the model in eval mode. Users may need to play with max iterations and learning rate to get convergeance.
 - Model predictions are provided for both the original and augmented image for comparison purposes.

```
import dropviz

corrupted_image, (model_output_original, model_output_corrupt) = dropviz.augment(<your_model>,
                                                                                 <your_model>.<some_layer>,
                                                                                 <device>,
                                                                                 <input_image>,
                                                                                 <max_iters>,
                                                                                 <tolerance>,
                                                                                 <learning_rate>
                                                                                )
```