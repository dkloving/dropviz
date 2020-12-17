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

corrupted_image, (model_output_original, model_output_corrupt) = dropviz.augment(model=<your_model>,
                                                                                 layer=<your_model>.<some_layer>,
                                                                                 device=<device>,
                                                                                 data=<input_image>,
                                                                                 n=<num_samples>,
                                                                                 max_epochs=<max_iters>,
                                                                                 loss_tolerance=<tolerance>,
                                                                                 lr=<learning_rate>,
                                                                                 verbosity=<optional>
                                                                                )
plt.imshow(corrupted_image[0])
plt.title('First Augmented Image (1 of <num_samples>)')
```

Parameters:
 - `model` a pytorch model that has layers which respond to `.train()` and `.eval()` modes.
 - `layer` the layer from which to collect activations.
 - `device` device to be used by pytorch, probably `cpu` or `cuda` for most users.
 - `data` a batch of data that can be passed through the model.
 - `n` number of samples to compute for a given image. Higher values will take more time.
 - `max_epochs` the algorithm will stop after this many iterations even if it hasn't converged. `10000`s or `1000000`s may be required for good results.
 - `loss_tolerance` the algorithm will stop if it reaches this MSE between dropout and no-dropout activations. Requires experimentation. Too high will not produce good results, too low will take too long.
 - `lr` the learning rate for the optimizer. Use much lower values than when training. `0.00001` ow less is not unreasonale. Requires experimentation.
 - `verbosity` set to 1 when tuning the above parameters to get more feedback on the optimization process.
