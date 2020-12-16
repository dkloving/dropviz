import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def get_activation(variable):
    """Callback to store the activation of a layer. Assumes listlike variable for scope purposes.

    Thanks to `ptrblck` at:
        https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/4
    """
    def hook(model, input, output):
        variable[0] = output
    return hook


def augment(model, layer, device, data, n, max_epochs, loss_tolerance, lr):

    # copy and send the data and model to the device
    data = data.detach().clone()
    data = data.repeat(n, 1, 1, 1).to(device)
    model.to(device)

    # set up intermediate activation
    activation = [None]  # this is just a hack to keep track of the tensor accross scopes
    activation_hook = layer.register_forward_hook(get_activation(activation))

    # get activation from forward pass with dropout
    model.train()
    original_output = model(data)
    activation_dropout = activation[0].detach().clone()

    # Set requires_grad attribute of data so we can corrupt it
    data.requires_grad = True

    # turn off dropout
    model.eval()

    # create optimizer
    optimizer = torch.optim.Adam([data], lr=lr)

    loss = 1.0
    ep = 0
    while (loss > loss_tolerance) and (ep < max_epochs):

        # forward pass the data through the model
        output = model(data)
        activation_output = activation[0]

        # calculate loss with respect to target
        loss = F.mse_loss(activation_output, activation_dropout)
        loss.backward()

        # update data
        optimizer.step()

        if ep % 1000 == 0:
            print(f"Epoch {ep+1}, loss {loss}")
        ep += 1
    print(f"Final loss {loss}")

    # clean up and remove activation hook
    activation_hook.remove()

    return data.squeeze().detach().cpu().numpy(), (original_output.squeeze().detach().cpu().numpy(),
                                                   output.squeeze().detach().cpu().numpy()
                                                   )
