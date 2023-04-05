"""
m6A_calling.py
Author: Anupama Jha <anupamaj@uw.edu>
Training and data generator code adapted from bpnet-lite (https://github.com/jmschrei/bpnet-lite)

This module contains a reference implementation of a CNN model for m6A calling.
GS hackathon participants: you can start with this code to develop your own
models for m6A calling.
"""

import torch
import argparse
import numpy as np
from torchsummary import summary
from sklearn.metrics import (confusion_matrix,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             balanced_accuracy_score,
                             matthews_corrcoef,
                             roc_auc_score,
                             average_precision_score,
                             roc_curve,
                             precision_recall_curve, 
                             auc
                             )


class DataGenerator(torch.utils.data.Dataset):
    """
    Data generator for the m6A model. It randomly selects
    batches from the data to train the m6A model.
    """
    def __init__(self, features, labels, random_state=None):
        """
        Constructor for the data generator class, expects features
        and labels matrices.
        :param features: numpy.array, Nx15, N=number of sequences,
                                      each sequence is of length 15.
        :param labels: numpy array, one hot encoded labels for whether
                                    a sequence in features variable contains
                                    methylated A or not.
        :param random_state: numpy.random_state, allow reproducibility by selecting
                                                 a seed for random operations beforehand.
        """
        self.random_state = random_state
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Get random indices from the training data
        to form batches.
        :param idx: numpy.array, indices to retrieve, ignoring for now.
        :return: x, y: features and labels from selected indices.
        """
        i = self.random_state.choice(len(self.features))

        x = self.features[i]
        y = self.labels[i]

        x = torch.tensor(x)
        y = torch.tensor(y)

        return x, y


def m6AGenerator(train_path, val_path, input_size, random_state=None, pin_memory=True,
                 num_workers=0, batch_size=32):
    """
    This generator returns a training data generator as well as
    validation features and labels
    :param train_path: str, path where the training data is stored.
    :param val_path: str, path where the val data is stored.
    :param random_state: int, seed for numpy random_state.
    :param pin_memory: bool, Makes CUDA efficient by skipping one copy
                             operation, true by default.
    :param num_workers: int, number of worker threads the generator can
                             utilize, 0 by default.
    :param batch_size: int, number of examples in each batch
    :return: X_gen: training data generator, X_val: validation features, y_val: validation labels
    """
    # initialize Random state
    random_state = np.random.RandomState(random_state)

    # Load training data
    train_data = np.load(train_path, allow_pickle=True)
    
    # Load training and validation features and labels
    # Sometimes we want to train on input subsets, this will 
    # achieve that. 
    X_train = train_data["features"][:, 0:input_size, :]
    y_train = train_data["labels"]
    
    y_train_ohe = np.zeros((len(y_train), 2))
    y_train_ohe[np.where(y_train == 1)[0], 0] = 1
    y_train_ohe[np.where(y_train == 0)[0], 1] = 1
    
    # Load validation data
    val_data = np.load(val_path, allow_pickle=True)
    X_val = val_data['features'][:, 0:input_size, :]
    y_val = val_data['labels']
    
    y_val_ohe = np.zeros((len(y_val), 2))
    y_val_ohe[np.where(y_val == 1)[0], 0] = 1
    y_val_ohe[np.where(y_val == 0)[0], 1] = 1

    print(f"Training features shape {X_train.shape}, training labels shape: {y_train.shape}")
    print(f"Validation features shape {X_val.shape}, validation labels shape: {y_val.shape}")

    # Get the training data generator
    X_gen = DataGenerator(X_train,
                          y_train_ohe,
                          random_state=random_state)

    # Wrap it in a data loader
    X_gen = torch.utils.data.DataLoader(X_gen,
                                        pin_memory=pin_memory,
                                        num_workers=num_workers,
                                        batch_size=batch_size, 
                                        shuffle=True)

    return X_gen, (X_val, y_val_ohe)


class M6ANet(torch.nn.Module):
    def __init__(self, input_size=7, n_layers=3, sec_last_layer_size=25, last_layer_size=5,
                 output_shape=2, model_name="m6ANet"):
        """
        Constructor for the M6ANet, a CNN model for m6A calling.
        :param input_size: int, number of channels in the data, currently 6, 4 for nucleotide identity, one for inter-
                                pulse distance and one for pulse width.
        :param n_layers: int, number of layers in the CNN model.
        :param sec_last_layer_size: int, size of the second last dense layer.
        :param last_layer_size: int, size of the last dense layer.
        :param output_shape: int, number of outputs, two in our case, m6A or not.
        """
        super(M6ANet, self).__init__()

        self.n_layers = n_layers

        # Assign a name to the model
        self.name = f"{model_name}.{n_layers}"

        # Three convolution layers with ReLU activation
        self.conv_1 = torch.nn.Conv1d(input_size, 30, kernel_size=5, stride=1)

        self.conv_2 = torch.nn.Conv1d(30, 10, kernel_size=5, stride=1)

        self.conv_3 = torch.nn.Conv1d(10, 5, kernel_size=3, stride=1)

        # a dense layer with ReLU activation
        self.linear = torch.nn.Linear(sec_last_layer_size, last_layer_size)

        # an output dense layer with no activation
        self.label = torch.nn.Linear(last_layer_size, output_shape)

        # Loss function
        self.cross_entropy_loss = torch.nn.BCELoss(reduction='mean')

    def forward(self, X):
        """
        Forward function to go from input to output
        of the model.
        :param X: Tensor, input to the model.
        :return: y: tensor, output from the model.
        """
        # Three convolutional layers with ReLU activation
        X = torch.nn.ReLU()(self.conv_1(X))
        X = torch.nn.ReLU()(self.conv_2(X))
        X = torch.nn.ReLU()(self.conv_3(X))
        # Condense 2D shape to 1D
        X = torch.flatten(X, 1)
        # Dense layer with ReLU activation
        X = torch.nn.ReLU()(self.linear(X))
        # Output layer
        y = torch.nn.Softmax(dim=1)(self.label(X))
        return y

    def predict(self, X, batch_size=64, device='cpu'):
        """
        Predict function to generate M6ANet model predictions.
        :param X: tensor, input data
        :param batch_size: int, batch size for generating predictions in.
        :return: m6a_labels: tensor, predictions
        """
        with torch.no_grad():
            # Get batch start indices
            starts = np.arange(0, X.shape[0], batch_size)
            # Get batch end indices
            ends = starts + batch_size
            # m6a labels
            m6a_labels = []
            # Get predictions for every batch
            for start, end in zip(starts, ends):
                X_batch = X[start:end].to(device)
                # Run the data through the forward
                # function to generate label predictions
                m6a_labels_ = self(X_batch)
                # Move the label predictions to the CPU
                m6a_labels_ = m6a_labels_.cpu()
                # Append to the list of all labels
                m6a_labels.append(m6a_labels_)
            # Make one list of all labels
            m6a_labels = torch.cat(m6a_labels)
            return m6a_labels

    def fit_generator(self, training_data, model_optimizer,
                      X_valid=None, y_valid=None, max_epochs=10,
                      verbose=True, validation_iter=1000, device='cpu'):
        """

        :param training_data: torch.DataLoader, training data generator
        :param model_optimizer: torch.Optimizer, An optimizer to training our model
        :param X_valid: numpy array, validation features
        :param y_valid: numpy array, validation labels
        :param max_epochs: int, maximum epochs to run the model for
        :param verbose: bool, whether to compute validation stats
        :param validation_iter: int, After how many iterations should we compute validation stats.
        :param device: str, GPU versus CPU, defaults to CPU
        :return: None
        """
        # Convert validation data into tensors
        X_valid = torch.tensor(X_valid).float()
        y_valid = torch.tensor(y_valid).float()
        X_valid = X_valid.to(device)
        y_valid = y_valid.to(device)

        best_auroc = 0
        for epoch in range(max_epochs):
            # to log cross-entropy loss to average over batches
            avg_train_loss = 0
            avg_train_iter = 0
            iteration = 0
            for data in training_data:
                # Get features and label batch
                X, y = data
                # Convert them to float
                X = X.float()
                y = y.float()
                X, y = X.to(device), y.to(device)

                # Clear the optimizer and set the model to training mode
                model_optimizer.zero_grad()
                self.train()

                # Run forward pass
                m6a_labels = self.forward(X)

                # Calculate the cross entropy loss
                cross_entropy_loss = self.cross_entropy_loss(m6a_labels, y)

                # Extract the cross entropy loss for logging
                cross_entropy_loss_ = cross_entropy_loss.item()

                # Do the back propagation
                cross_entropy_loss.backward()
                model_optimizer.step()

                # log loss to average over training batches
                avg_train_loss += cross_entropy_loss_
                avg_train_iter += 1

                # If verbose is true and current iteration is a validation iteration
                # compute validation stats.
                if verbose and iteration % validation_iter == 0:
                    with torch.no_grad():
                        # Set the model to evaluation mode
                        self.eval()
                        # Convert one hot encoded labels to number based labels ([0, 1] -> 1, [1, 0] -> 0)
                        y_valid_metric = torch.argmax(y_valid, dim=1).int()
                        # Compute the predictions for the validation set
                        valid_preds = self.predict(X_valid, device=device)
                        # Move predictions to CPU/GPU
                        valid_preds = valid_preds.to(device)
                        # Convert one hot encoded predictions to number based labels ([0, 1] -> 1, [1, 0] -> 0)
                        pred_valid_metric = torch.argmax(valid_preds, dim=1).int()
                        # compute cross_entropy loss for the validation set.
                        cross_entropy_loss = self.cross_entropy_loss(valid_preds, y_valid)
                        # Extract the validation loss
                        valid_loss = cross_entropy_loss.item()
                        
                        # Compute AUROC
                        sklearn_rocauc = roc_auc_score(y_valid.cpu().numpy()[:, 0], valid_preds.cpu().numpy()[:, 0])
                        
                        # Compute AUPR/Average precision
                        sklearn_ap = average_precision_score(y_valid.cpu().numpy()[:, 0], valid_preds.cpu().numpy()[:, 0])
                        
                        # Compute accuracy
                        sklearn_acc = accuracy_score(y_valid_metric.cpu().numpy(), pred_valid_metric.cpu().numpy())

                        print(f"Epoch {epoch}, iteration {iteration}, train loss: {(avg_train_loss / avg_train_iter):4.4f}, validation loss: {valid_loss:4.4f}")
                        print(f"Validation iteration {iteration}, AUPR: {sklearn_ap}, Accuracy: {sklearn_acc}, AUROC: {sklearn_rocauc}")

                        if sklearn_rocauc > best_auroc:
                            torch.save(self, "{}.best.torch".format(self.name))
                            best_auroc = sklearn_rocauc

                        avg_train_loss = 0
                        avg_train_iter = 0

                iteration += 1

        torch.save(self, "{}.final.torch".format(self.name))


def force_cudnn_initialization():
    """
    Force cuda to reinitialize. 
    """
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--train_data',
        type=str,
        default="/net/noble/vol4/noble/user/anupamaj/proj/m6A-calling/data/PS00075_1_2022-10-17.npz",
        help="path to the training npz file. Default is in the data directory"
    )
    
    parser.add_argument(
        '--val_data',
        type=str,
        default="/net/noble/vol4/noble/user/anupamaj/proj/m6A-calling/data/PS00075_2_2022-10-17.npz",
        help="path to the val npz file. Default is in the data directory"
    )
    
    parser.add_argument(
        '--pretrain_model',
        type=str,
        default="models/m6ANet_other_half_hifi.3.best.torch",
        help="Path of the model to be stored."
    )
    
    parser.add_argument(
        '--model_save_path',
        type=str,
        default="models/m6ANet_PS00075_no_init",
        help="Path of the model to be stored."
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help="Training on cpu or cuda. Default is cuda"
    )
    
    parser.add_argument(
        '--input_size',
        type=int,
        default=6,
        help="Input size."
    )

    args = parser.parse_args()
    

    print(f"Training on {args.device}")
    
    if args.device=='cuda':
        force_cudnn_initialization()
    
    #model = torch.load(args.pretrain_model).to(args.device)
    
    # Move the model to appropriate device
    model = M6ANet(input_size=args.input_size, model_name=args.model_save_path).to(args.device)
    #model.name = args.model_save_path
    
    # Adam optimizer with learning rate 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Print model architecture summary
    summary_str = summary(model, input_size=(args.input_size, 15))
    
    # Get training data generator and validation data. 
    X_train, (X_val, y_val) = m6AGenerator(args.train_data, args.val_data, input_size=args.input_size, 
                                           random_state=None, pin_memory=True, num_workers=2, batch_size=32)
    
    # Train the model
    model.fit_generator(X_train,
                        optimizer,
                        X_valid=X_val, 
                        y_valid=y_val,
                        max_epochs=30, 
                        validation_iter=300000,
                        device=args.device)
    
