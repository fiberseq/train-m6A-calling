"""
m6a_cnn.py
Author: Anupama Jha <anupamaj@uw.edu>
This module predicts whether an
adenine is methylated or not. The
model is trained with Fiber-seq
HiFi read sequence, inter-pulse
distance and pulse width signal
from pacbio. The model is a
convolution neural network.
"""

import torch
import numpy as np
import _pickle as pickle
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

verbose = False


class M6ANet(torch.nn.Module):
    def __init__(
            self, input_size=6, sec_last_layer_size=25, last_layer_size=5, output_shape=2
    ):
        """
        Constructor for the M6ANet, a CNN
        model for m6A calling.
        :param input_size: int, number of
                                channels in
                                the data,
                                currently 6,
                                4 for nucleotide
                                identity, one for
                                inter-pulse distance
                                and one for pulse width.
        :param sec_last_layer_size: int, size of the second
                                         last dense layer.
        :param last_layer_size: int, size of the last dense
                                     layer.
        :param output_shape: int, number of outputs, two in
                                  our case, m6A or not.
        """
        super(M6ANet, self).__init__()

        # Three convolution layers with ReLU activation
        self.conv_1 = torch.nn.Conv1d(
            in_channels=input_size, out_channels=30, kernel_size=5, stride=1
        )

        self.relu_1 = torch.nn.ReLU()

        self.conv_2 = torch.nn.Conv1d(
            in_channels=30, out_channels=10, kernel_size=5, stride=1
        )

        self.relu_2 = torch.nn.ReLU()

        self.conv_3 = torch.nn.Conv1d(
            in_channels=10, out_channels=5, kernel_size=3, stride=1
        )

        self.relu_3 = torch.nn.ReLU()

        # a dense layer with ReLU activation
        self.linear = torch.nn.Linear(
            in_features=sec_last_layer_size, out_features=last_layer_size
        )

        self.relu_4 = torch.nn.ReLU()

        # an output dense layer with no activation
        self.label = torch.nn.Linear(
            in_features=last_layer_size, out_features=output_shape
        )

        # Loss function
        self.cross_entropy_loss = torch.nn.BCELoss(reduction="mean")

    def forward(self, X):
        """
        Forward function to go
        from input to output
        of the model.
        :param X: Tensor, input to
                          the model.
        :return: y: tensor, output
                            from the
                            model.
        """
        # Three convolutional layers
        # with ReLU activation
        X = self.relu_1(self.conv_1(X))
        X = self.relu_2(self.conv_2(X))
        X = self.relu_3(self.conv_3(X))

        # Condense 2D shape to 1D
        X = torch.flatten(X, 1)

        # Dense layer with ReLU activation
        X = self.relu_4(self.linear(X))

        # Output layer
        y = torch.nn.Softmax(dim=1)(self.label(X))
        return y

    def predict(self, X, batch_size=64, device="cpu"):
        """
        Predict function to generate
        M6ANet model predictions.
        :param X: tensor, input data
        :param batch_size: int, batch
                                size for
                                generating
                                predictions in.
        :param device: str, cpu or cuda
        :return: m6a_labels: tensor, predictions
        """
        # Turn off gradient
        # computation
        with torch.no_grad():
            # set model to
            # evaluation mode
            self.eval()

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
                m6a_labels_batch = self(X_batch)

                # Move the label predictions to the CPU
                m6a_labels_batch = m6a_labels_batch.cpu()

                # Append to the list of all labels
                m6a_labels.append(m6a_labels_batch)

            # Make one list of all labels
            m6a_labels = torch.cat(m6a_labels)
            return m6a_labels

    def evaluate(self,
                 X_valid,
                 y_valid,
                 device="cpu"):
        """
        Generate predictions for validation
        data and compute average precision.
        :param X_valid: np.array, validation
                                  features
        :param y_valid: np.array, validation
                                  labels
        :param device: str, cpu or cuda
        :return: float, average precision
        """
        # Convert validation data into tensors
        X_valid = torch.tensor(X_valid).float()
        y_valid = torch.tensor(y_valid).float()
        X_valid = X_valid.to(device)
        y_valid = y_valid.to(device)

        with torch.no_grad():
            # Set the model to
            # evaluation mode
            self.eval()

            # Compute the predictions for
            # the validation set
            valid_preds = self.predict(X_valid, device=device)

            # Compute AUPR/Average precision
            sklearn_ap = average_precision_score(
                y_valid.cpu().numpy()[:, 0], valid_preds.cpu().numpy()[:, 0]
            )

        return sklearn_ap

    def fit_semisupervised(
            self,
            training_data,
            model_optimizer,
            X_valid=None,
            y_valid=None,
            max_epochs=10,
            validation_iter=1000,
            device="cpu",
            best_save_model="",
            final_save_model="",
            prev_aupr=0,
            input_example=(1, 6, 15)
    ):
        """
        Training procedure for the semi-supervised version
        of m6A CNN.
        :param training_data: torch.DataLoader,
                              training data generator
        :param model_optimizer: torch.Optimizer,
                                An optimizer to
                                training our model
        :param X_valid: numpy array, validation features
        :param y_valid: numpy array, validation labels
        :param max_epochs: int, maximum epochs to run
                                the model for
        :param validation_iter: int,After how many
                                    iterations should
                                    we compute validation
                                    stats.
        :param device: str, GPU versus CPU, defaults to CPU
        :param best_save_model: str, path to save best model
        :param final_save_model: str, path to save final model
        :param prev_aupr: float, best precision so far,
                                 relevant for semi-supervised
                                 training
        :return: None
        """
        # Convert validation data into tensors
        X_valid = torch.tensor(X_valid).float()
        y_valid = torch.tensor(y_valid).float()
        X_valid = X_valid.to(device)
        y_valid = y_valid.to(device)

        best_aupr = prev_aupr
        for epoch in range(max_epochs):
            # to log cross-entropy loss to
            # average over batches
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

                # Clear the optimizer and
                # set the model to training mode
                model_optimizer.zero_grad()
                self.train()

                # Run forward pass
                m6a_labels = self.forward(X)

                # Calculate the cross entropy loss
                cross_entropy_loss = self.cross_entropy_loss(m6a_labels, y)

                # Extract the cross entropy loss for logging
                cross_entropy_loss_item = cross_entropy_loss.item()

                # Do the back propagation
                cross_entropy_loss.backward()
                model_optimizer.step()

                # log loss to average over training batches
                avg_train_loss += cross_entropy_loss_item
                avg_train_iter += 1

                # If current iteration is a
                # validation iteration
                # compute validation stats.
                if iteration > 0 and iteration % validation_iter == 0:
                    with torch.no_grad():
                        # Set the model to
                        # evaluation mode
                        self.eval()

                        # Convert one hot encoded labels
                        # to number based labels
                        # ([0, 1] -> 1, [1, 0] -> 0)
                        y_valid_metric = torch.argmax(y_valid, dim=1).int()

                        # Compute the predictions for
                        # the validation set
                        valid_preds = self.predict(X_valid, device=device)
                        # Move predictions to CPU/GPU
                        valid_preds = valid_preds.to(device)

                        # Compute AUPR/Average precision
                        sklearn_ap = average_precision_score(
                            y_valid.cpu().numpy()[:, 0], valid_preds.cpu().numpy()[:, 0]
                        )
                        if verbose:
                            # Convert one hot encoded predictions
                            # to number based labels
                            # ([0, 1] -> 1, [1, 0] -> 0)
                            pred_valid_metric = torch.argmax(valid_preds, dim=1).int()

                            # compute cross_entropy loss
                            # for the validation set.
                            cross_entropy_loss = self.cross_entropy_loss(
                                valid_preds, y_valid
                            )

                            # Extract the validation loss
                            valid_loss = cross_entropy_loss.item()
                            # Compute AUROC
                            sklearn_rocauc = roc_auc_score(
                                y_valid.cpu().numpy()[:, 0],
                                valid_preds.cpu().numpy()[:, 0],
                            )
                            # Compute accuracy
                            sklearn_acc = accuracy_score(
                                y_valid_metric.cpu().numpy(),
                                pred_valid_metric.cpu().numpy(),
                            )
                            train_loss = avg_train_loss / avg_train_iter
                            print(
                                f"Epoch {epoch}, iteration {iteration},"
                                f" train loss: {train_loss:4.4f},"
                                f" validation loss: {valid_loss:4.4f}"
                            )

                            print(
                                f"Validation iteration {iteration}, "
                                f"AUPR: {sklearn_ap},"
                                f" Accuracy: {sklearn_acc}, "
                                f"AUROC: {sklearn_rocauc}"
                            )

                        if sklearn_ap > best_aupr:
                            with open(best_save_model, "wb") as fp:
                                pickle.dump(self.state_dict(), fp)
                            best_aupr = sklearn_ap
                            # save rust model
                            example = torch.rand(input_example).float().to(device)
                            traced_script_module = torch.jit.trace(self, example)
                            traced_script_module.save(f"{best_save_model}.pt")

                        avg_train_loss = 0
                        avg_train_iter = 0

                iteration += 1

        with open(final_save_model, "wb") as fp:
            pickle.dump(self.state_dict(), fp)
        # save rust model
        example = torch.rand(input_example).float().to(device)
        traced_script_module = torch.jit.trace(self, example)
        traced_script_module.save(f"{final_save_model}.pt")
        

    def fit_supervised(
            self,
            training_data,
            model_optimizer,
            X_valid=None,
            y_valid=None,
            max_epochs=10,
            validation_iter=1000,
            device="cpu",
            best_save_model="",
            final_save_model="",
            input_example=(1, 6, 15)
    ):
        """
        Training procedure for the supervised version
        of m6A CNN.
        :param training_data: torch.DataLoader,
                              training data generator
        :param model_optimizer: torch.Optimizer,
                                An optimizer to
                                training our model
        :param X_valid: numpy array, validation features
        :param y_valid: numpy array, validation labels
        :param max_epochs: int, maximum epochs to run
                                the model for
        :param validation_iter: int,After how many
                                    iterations should
                                    we compute validation
                                    stats.
        :param device: str, GPU versus CPU, defaults to CPU
        :param best_save_model: str, path to save best model
        :param final_save_model: str, path to save final model
        :return: None
        """
        # Convert validation data into tensors
        X_valid = torch.tensor(X_valid).float()
        y_valid = torch.tensor(y_valid).float()
        X_valid = X_valid.to(device)
        y_valid = y_valid.to(device)

        best_aupr = 0
        for epoch in range(max_epochs):
            # to log cross-entropy loss to
            # average over batches
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

                # Clear the optimizer and
                # set the model to training mode
                model_optimizer.zero_grad()
                self.train()

                # Run forward pass
                m6a_labels = self.forward(X)

                # Calculate the cross entropy loss
                cross_entropy_loss = self.cross_entropy_loss(m6a_labels, y)

                # Extract the cross entropy loss for logging
                cross_entropy_loss_item = cross_entropy_loss.item()

                # Do the back propagation
                cross_entropy_loss.backward()
                model_optimizer.step()

                # log loss to average over training batches
                avg_train_loss += cross_entropy_loss_item
                avg_train_iter += 1

                # If current iteration is a
                # validation iteration
                # compute validation stats.
                if iteration % validation_iter == 0:
                    with torch.no_grad():
                        # Set the model to
                        # evaluation mode
                        self.eval()

                        # Convert one hot encoded labels
                        # to number based labels
                        # ([0, 1] -> 1, [1, 0] -> 0)
                        y_valid_metric = torch.argmax(y_valid, dim=1).int()

                        # Compute the predictions for
                        # the validation set
                        valid_preds = self.predict(X_valid, device=device)
                        # Move predictions to CPU/GPU
                        valid_preds = valid_preds.to(device)

                        # Convert one hot encoded predictions
                        # to number based labels
                        # ([0, 1] -> 1, [1, 0] -> 0)
                        pred_valid_metric = torch.argmax(valid_preds, dim=1).int()

                        # compute cross_entropy loss
                        # for the validation set.
                        cross_entropy_loss = self.cross_entropy_loss(
                            valid_preds, y_valid
                        )

                        # Extract the validation loss
                        valid_loss = cross_entropy_loss.item()

                        # Compute AUROC
                        sklearn_rocauc = roc_auc_score(
                            y_valid.cpu().numpy()[:, 0], valid_preds.cpu().numpy()[:, 0]
                        )

                        # Compute AUPR/Average precision
                        sklearn_ap = average_precision_score(
                            y_valid.cpu().numpy()[:, 0], valid_preds.cpu().numpy()[:, 0]
                        )

                        # Compute accuracy
                        sklearn_acc = accuracy_score(
                            y_valid_metric.cpu().numpy(),
                            pred_valid_metric.cpu().numpy(),
                        )
                        train_loss = avg_train_loss / avg_train_iter

                        print(
                            f"Epoch {epoch}, iteration {iteration},"
                            f" train loss: {train_loss:4.4f},"
                            f" validation loss: {valid_loss:4.4f}"
                        )

                        print(
                            f"Validation iteration {iteration}, "
                            f"AUPR: {sklearn_ap},"
                            f" Accuracy: {sklearn_acc}, "
                            f"AUROC: {sklearn_rocauc}"
                        )

                        if sklearn_ap > best_aupr:
                            with open(best_save_model, "wb") as fp:
                                pickle.dump(self.state_dict(), fp)
                            best_aupr = sklearn_ap
                            # save rust model
                            example = torch.rand(input_example).float().to(device)
                            traced_script_module = torch.jit.trace(self, example)
                            traced_script_module.save(f"{best_save_model}.pt")

                        avg_train_loss = 0
                        avg_train_iter = 0

                iteration += 1

        with open(final_save_model, "wb") as fp:
            pickle.dump(self.state_dict(), fp)
        
        # save rust model
        example = torch.rand(input_example).float().to(device)
        traced_script_module = torch.jit.trace(self, example)
        traced_script_module.save(f"{final_save_model}.pt")
