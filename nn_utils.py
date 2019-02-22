import numpy as np
import torch
from torch.utils import data 
from torchvision import transforms
import torch.nn as nn



"""
Utility functions used for my imitation learning.
"""


# This is most likely not best practice...
torch.set_default_dtype(torch.float32)



def make_histories(states, history_length, sampling_sparsity=1):
    """Used to make a dataset suitable for a neural network who's state is a history of inputs.


    This function takes numpy array states which should be a time series, and returns an array of histories of size
    (history_length, num_states) the optional sampling_sparsity parameter decides how many time steps to look back for
    every entry in a history. This is probably best explained by looking at the return value:

    histories[i] = np.array([states[i], states[i - T], states[i - 2*T], ... ])

    Attributes:
        states:  input numpy array, must be 2 dimensional (num_samples, num_states)
        history_length:  number of samples in each history
        sampling_sparsity:  timesteps between samples in each history

    Returns:
        histories: numpy array (num_samples, num_states, history_length)

    Example:
        states = np.random.randn(12,2)
        history_state = make_histories(states, 3)


      """

    num_set = states.shape[0]
    z_ext = np.zeros(((history_length - 1) * sampling_sparsity, states.shape[1]))
    states = np.concatenate((z_ext, states), axis=0)
    histories = np.zeros((num_set,) + (states.shape[1],) + (history_length,)) # initialize output matrix
    step = 0

    while(step<num_set):
        # select vectors according to history_length and sampling_sparsity
        histories[step, :, :] = np.transpose(states[step:(history_length - 1) * sampling_sparsity + 1 + step:sampling_sparsity, :])
        step+=1
    return histories


def fit_model(model, state_train, action_train, num_epochs, learning_rate = 1e-2, batch_size=32, shuffle=True):
    """
    Trains a pytorch module model to predict actions from states for num_epochs passes through the dataset.

    This is used to do a (relatively naive) version of behavior cloning
    pretty naive (but fully functional) training loop right now, will want to keep adding to this and will want to
    eventually make it more customizable.

    The hope is that this will eventually serve as a keras model.fit funtion, but custimized to our needs.


    Attributes:
        model: pytorch module implementing your controller
        states_train numpy array (or pytorch tensor) of states (inputs to your network) you want to train over
        action_train: numpy array (or pytorch tensor) of actions (outputs of the network)
        num_epochs: how many passes through the dataset to make
        learning_rate: initial learning rate for the adam optimizer

    Returns:
        Returns a list of average losses per epoch
        but note that the model is trained in place!!


    Example:
        model = nn.Sequential(
            nn.Linear(4,12),
            nn.ReLU(),
            nn.Linear(12,12),
            nn.ReLU(),
            nn.Linear(12,1)
            )

        states = np.random.randn(100,4)
        actions = np.random.randn(100,1)

        loss_hist = fit_model(model,states, actions, 200)


    """
    # Check if GPU is available , else fall back to CPU
    # TODO this might belong in module body
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Normalize training data set
    state_train_norm, state_train_mean, state_train_std = normalize_data(state_train)
    action_train_norm, action_train_mean, action_train_std = normalize_data(action_train)
    
    state_tensor = torch.as_tensor(state_train_norm, dtype = torch.float32) # make sure that our input is a tensor
    action_tensor = torch.as_tensor(action_train_norm, dtype = torch.float32)

    training_data = data.TensorDataset(state_tensor, action_tensor)
    training_generator = data.DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)

    # action_size = action_train.size()[1]

    loss_hist = []
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0

        for local_states, local_actions in training_generator:

            # Transfer to GPU (if GPU is enabled, else this does nothing)
            local_states, local_actions = local_states.to(device), local_actions.to(device)

            # predict and calculate loss for the batch
            action_preds = model(local_states)
            loss = loss_fn(local_actions, action_preds)
            epoch_loss += loss # only used for metrics

            # do the normal pytorch update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # after each epoch append the average loss
        loss_hist.append(epoch_loss / len(state_train))

    return loss_hist


def normalize_data(q):
    """
    Normalizes a data set to 0 mean and 1 variance.

    This function takes numpy array of states, which should be a time series,
    and returns a normalized version of the states along with the mean and 
    standard deviation of the state array.


    Attributes:
        q:  input numpy array of state trajectories

    Returns:
        q_norm: output numpy array of normalized states
        q_mean: numpy array of mean of each state trajectory
        q_std: numpy array of stddev of each state trajectory
        
    Example:
        q_norm, q_mean, q_std = normalize_data(q)

      """
      
    q_std = np.std(q)
    q_mean = np.mean(q)
    
    q_norm = (q-q_mean)/q_std
    
    return q_norm, q_mean, q_std