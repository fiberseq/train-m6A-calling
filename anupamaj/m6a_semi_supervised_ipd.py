"""
m6A_semi_supervised.py
Author: Anupama Jha <anupamaj@uw.edu>
Training and data generator code adapted from bpnet-lite
(https://github.com/jmschrei/bpnet-lite)

This module contains an implementation of a semi-supervised
CNN model for m6A calling. We implement a similar approach
to the percolator algorithm for a CNN model. In the first
iteration, we rank m6A calls using the IPD value of the
central nucleotide, we threshold this rank list at the
given FDR threshold (ideally 1%). We feed this positive
set along with the  all the negative data to our CNN model,
after training for an iteration, we re-rank the the positive
and negative data and repeat for x iterations. We compute the
FDR on the validation set using the CNN prediction on the
validation set after every iteration, we stop after we reach
the targeted FDR rate on the validation data.
"""

import torch
import argparse
import numpy as np
import xgboost as xgb
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


def count_pos_neg(labels, set_name=""):
    """
    Count the number of positive (m6A) and negative (everything else) examples in our set
    :param labels: np.array, one-hot-encoded label
    :param set_name: str, training, validation or test set.
    :return:
    """
    # First column are positive labels
    m6as = np.where(labels == 1)[0]
    # Second column are negative labels
    nulls = np.where(labels == 0)[0]
    num_pos = len(m6as)
    num_neg = len(nulls)
    print(f"{set_name} has {num_pos} positives and {num_neg} negatives")
    return num_pos, num_neg


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

    # Take all train labels
    rand_train = np.random.choice(np.arange(len(y_train), dtype=int),
                                  size=(int(0.05 * len(y_train)),), replace=False)

    X_train = X_train[rand_train, :, :]
    y_train = y_train[rand_train]

    # Load validation data
    val_data = np.load(val_path, allow_pickle=True)
    X_val = val_data['features'][:, 0:input_size, :]
    y_val = val_data['labels']

    # Take 50% val labels
    rand_val = np.random.choice(np.arange(len(y_val), dtype=int),
                                size=(int(0.01 * len(y_val)),), replace=False)

    X_val = X_val[rand_val, :, :]
    y_val = y_val[rand_val]

    print(f"Training features shape {X_train.shape}, training labels shape: {y_train.shape}")
    print(f"Validation features shape {X_val.shape}, validation labels shape: {y_val.shape}")

    count_pos_neg(y_train, set_name="Train")
    count_pos_neg(y_val, set_name="Validation")

    return X_train, y_train, X_val, y_val, random_state


def _fdr2qvalue(fdr, num_total, met, indices):
    """
    Quickly turn a list of FDRs to q-values.
    All of the inputs are assumed to be sorted.
    Parameters
    ----------
    fdr : numpy.ndarray
        A vector of all unique FDR values.
    num_total : numpy.ndarray
        A vector of the cumulative number of PSMs at each score.
    met : numpy.ndarray
        A vector of the scores for each PSM.
    indices : tuple of numpy.ndarray
        Tuple where the vector at index i indicates the PSMs that
        shared the unique FDR value in `fdr`.
    Returns
    -------
    numpy.ndarray
        A vector of q-values.
    """
    min_q = 1
    qvals = np.ones(len(fdr))
    group_fdr = np.ones(len(fdr))
    prev_idx = 0
    for idx in range(met.shape[0]):
        next_idx = prev_idx + indices[idx]
        group = slice(prev_idx, next_idx)
        prev_idx = next_idx

        fdr_group = fdr[group]
        n_group = num_total[group]
        curr_fdr = fdr_group[np.argmax(n_group)]
        if curr_fdr < min_q:
            min_q = curr_fdr

        group_fdr[group] = curr_fdr
        qvals[group] = min_q

    return qvals


def tdc(scores, target, pn_ratio=2, desc=True):
    """
    Method from Will Fondrie's mokapot, remember to cite:
    https://github.com/wfondrie/mokapot/blob/master/mokapot/qvalues.py

    Estimate q-values using target decoy competition.
    Estimates q-values using the simple target decoy competition method.
    For set of target and decoy PSMs meeting a specified score threshold,
    the false discovery rate (FDR) is estimated as:
    ...math:
        FDR = \frac{2*Decoys}{Targets + Decoys}
    More formally, let the scores of target and decoy PSMs be indicated as
    :math:`f_1, f_2, ..., f_{m_f}` and :math:`d_1, d_2, ..., d_{m_d}`,
    respectively. For a score threshold :math:`t`, the false discovery
    rate is estimated as:
    ...math:
        E\\{FDR(t)\\} = \frac{|\\{d_i > t; i=1, ..., m_d\\}| + 1}
        {\\{|f_i > t; i=1, ..., m_f|\\}}
    The reported q-value for each PSM is the minimum FDR at which that
    PSM would be accepted.
    Parameters
    ----------
    scores : numpy.ndarray of float
        A 1D array containing the score to rank by
    target : numpy.ndarray of bool
        A 1D array indicating if the entry is from a target or decoy
        hit. This should be boolean, where `True` indicates a target
        and `False` indicates a decoy. `target[i]` is the label for
        `metric[i]`; thus `target` and `metric` should be of
        equal length.
    pn_ratio: float, ratio of positive/negative class examples
    desc : bool
        Are higher scores better? `True` indicates that they are,
        `False` indicates that they are not.
    Returns
    -------
    numpy.ndarray
        A 1D array with the estimated q-value for each entry. The
        array is the same length as the `scores` and `target` arrays.
    """
    scores = np.array(scores)

    try:
        target = np.array(target, dtype=bool)
    except ValueError:
        raise ValueError("'target' should be boolean.")

    if scores.shape[0] != target.shape[0]:
        raise ValueError("'scores' and 'target' must be the same length")

    # Unsigned integers can cause weird things to happen.
    # Convert all scores to floats to for safety.
    if np.issubdtype(scores.dtype, np.integer):
        scores = scores.astype(np.float_)

    # Sort and estimate FDR
    if desc:
        srt_idx = np.argsort(-scores)
    else:
        srt_idx = np.argsort(scores)

    scores = scores[srt_idx]
    target = target[srt_idx]
    cum_targets = target.cumsum()
    cum_decoys = ((target - 1) ** 2).cumsum()
    num_total = cum_targets + cum_decoys

    # Handles zeros in denominator
    fdr = np.divide(
        (1 + pn_ratio) * cum_decoys,
        (cum_targets + cum_decoys),
        out=np.ones_like(cum_targets, dtype=float),
        where=(cum_targets != 0),
    )

    # Calculate q-values
    unique_metric, indices = np.unique(scores, return_counts=True)

    # Some arrays need to be flipped so that we can loop through from
    # worse to best score.
    fdr = np.flip(fdr)
    num_total = np.flip(num_total)
    if not desc:
        unique_metric = np.flip(unique_metric)
        indices = np.flip(indices)

    qvals = _fdr2qvalue(fdr, num_total, unique_metric, indices)
    qvals = np.flip(qvals)
    qvals = qvals[np.argsort(srt_idx)]

    return qvals


def compute_pos_neg_sets(x_data, scores, y_data, score_threshold):
    """
    Compute positive and negative sets using the validation
    score threshold.
    :param x_data: np.array, features are stored here.
    :param scores: np.array, CNN scores for data.
    :param y_data: np.array, m6A labels.
    :param score_threshold: float, score threshold based on
                                validation data
    :return: new features, new labels and all labels.
    """
    pos_set_score = np.where(scores >= score_threshold)[0]
    pos_set_all = np.where(y_data == 1)[0]
    pos_set = np.intersect1d(pos_set_score, pos_set_all)
    neg_set_all = np.where(y_data == 0)[0]
    neg_set = neg_set_all

    y_data_init = np.zeros((len(pos_set) + len(neg_set),))
    y_data_init[0:len(pos_set)] = 1

    x_data_init = np.concatenate((x_data[pos_set, :, :], x_data[neg_set, :, :]))

    shuffle_idx = np.arange(len(y_data_init), dtype=int)
    np.random.shuffle(shuffle_idx)

    x_data_init = x_data_init[shuffle_idx, :, :]
    y_data_init = y_data_init[shuffle_idx]
    count_pos_neg(y_data_init, set_name="New FDR set")
    return x_data_init, y_data_init


def compute_fdr_score(scores, y_data, fdr_threshold=0.1):
    """
    Compute FDR score threshold using the validation data
    :param scores: np.array, array of xgboost/CNN scores
    :param y_data: np.array, ground truth labels
    :param fdr_threshold: float, false discovery rate
    :return:
    """
    num_pos_class = len(np.where(y_data == 1)[0])
    num_neg_class = len(np.where(y_data == 0)[0])
    pn_ratio = num_pos_class / float(num_neg_class)
    print(f"pn_ratio: {pn_ratio}")
    ipd_fdr = tdc(scores, y_data, pn_ratio, desc=True)
    print(f"ipd_fdr: min: {np.min(ipd_fdr)}, max: {np.max(ipd_fdr)}, "
          f"mean: {np.mean(ipd_fdr)}, median: {np.median(ipd_fdr)}, std: {np.std(ipd_fdr)}")
    pos_set = np.where(ipd_fdr <= fdr_threshold)[0]
    
    if len(pos_set) == 0:
        pos_set = np.where(ipd_fdr <= np.min(ipd_fdr))[0]
        
    if torch.is_tensor(pos_set):  
        pos_set = pos_set.numpy()
        
    if torch.is_tensor(scores):  
        scores = scores.numpy()
        
    #print(f"pos_set: {pos_set}")
    pos_scores = scores[pos_set]
    #print(f"pos_scores: {pos_scores}")
    score_thresholds = np.min(pos_scores)
    num_pos = len(pos_scores)

    return score_thresholds, num_pos


class M6ANet(torch.nn.Module):
    def __init__(self, input_size=7, n_layers=3, sec_last_layer_size=25, last_layer_size=5,
                 output_shape=2, model_name="m6ANet"):
        """
        Constructor for the M6ANet, a CNN model for m6A calling.
        :param input_size: int, number of channels in the data, currently 6,
                                4 for nucleotide identity, one for inter-
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
                      verbose=True, validation_iter=1000, device='cpu', prev_val_precision=0):
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

        best_auroc = prev_val_precision
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
                        sklearn_ap = average_precision_score(y_valid.cpu().numpy()[:, 0],
                                                             valid_preds.cpu().numpy()[:, 0])

                        # Compute accuracy
                        sklearn_acc = accuracy_score(y_valid_metric.cpu().numpy(), pred_valid_metric.cpu().numpy())

                        print(f"Epoch {epoch}, iteration {iteration}, "
                              f"train loss: {(avg_train_loss / avg_train_iter):4.4f}, "
                              f"validation loss: {valid_loss:4.4f}")
                        print(f"Validation iteration {iteration}, AUPR: {sklearn_ap}, "
                              f"Accuracy: {sklearn_acc}, AUROC: {sklearn_rocauc}")

                        if sklearn_rocauc > best_auroc:
                            torch.save(self, "{}.best.torch".format(self.name))
                            best_auroc = sklearn_rocauc

                        avg_train_loss = 0
                        avg_train_iter = 0

                iteration += 1

        torch.save(self, "{}.final.torch".format(self.name))
        return sklearn_ap


def force_cudnn_initialization():
    """
    Force cuda to reinitialize.
    """
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev),
                               torch.zeros(s, s, s, s, device=dev))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_data',
        type=str,
        default="/net/noble/vol4/noble/user/anupamaj/proj/m6A-calling/data/PS00075_2.npz",
        help="path to the training npz file. Default is in the data directory"
    )

    parser.add_argument(
        '--val_data',
        type=str,
        default="/net/noble/vol4/noble/user/anupamaj/proj/m6A-calling/data/PS00075_3.npz",
        help="path to the val npz file. Default is in the data directory"
    )

    parser.add_argument(
        '--save_pos',
        type=str,
        default="results/val_pos_identified",
        help="path to the val npz file. Default is in the data directory"
    )

    parser.add_argument(
        '--pretrain_model',
        type=str,
        default="../mvollger/models/xgboost_2022-10-17.0.81.bin",
        help="Path of the model to be stored."
    )

    parser.add_argument(
        '--model_load_path',
        type=str,
        default="models/m6ANet_PS00075_no_init.3.best_supervised.torch",
        help="Path of the model to be stored."
    )
    
    parser.add_argument(
        '--model_save_path',
        type=str,
        default="models/m6ANet_PS00075_semi_supervised",
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

    if args.device == 'cuda':
        force_cudnn_initialization()

    all_num_pos = []
    val_ap = []

    # Move the model to appropriate device
    #model = M6ANet(input_size=args.input_size,
    #               model_name=args.model_save_path).to(args.device)
    
    model = torch.load(args.model_load_path, map_location=torch.device(args.device))
    model.model_name = args.model_save_path
    

    # Adam optimizer with learning rate 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Print model architecture summary
    summary_str = summary(model, input_size=(args.input_size, 15))

    # Get training data generator and validation data.
    X_train, y_train, X_val, y_val, random_state = m6AGenerator(args.train_data,
                                                                args.val_data,
                                                                input_size=args.input_size,
                                                                random_state=None,
                                                                pin_memory=True,
                                                                num_workers=2,
                                                                batch_size=32)

    y_score_val = X_val[:, 4, 7]

    y_val_ohe = np.zeros((len(y_val), 2))
    y_val_ohe[np.where(y_val == 1)[0], 0] = 1
    y_val_ohe[np.where(y_val == 0)[0], 1] = 1

    score_threshold, num_pos = compute_fdr_score(y_score_val,
                                                 np.array(y_val, dtype=bool),
                                                 fdr_threshold=0.05)
    

    all_num_pos.append(num_pos)

    sklearn_ap = average_precision_score(y_val,
                                         y_score_val)

    print(f"Validation IPD average precision: {sklearn_ap}, "
          f" Number of positives at FDR of 5% are: {num_pos}")

    val_ap.append(sklearn_ap)

    y_score = X_train[:, 4, 7]

    print(f"y_score: {y_score.shape}, min: {np.min(y_score)}, "
          f"max: {np.max(y_score)}, mean: {np.mean(y_score)}, std: {np.std(y_score)}")

    y_train_ohe = np.zeros((len(y_train), 2))
    y_train_ohe[np.where(y_train == 1)[0], 0] = 1
    y_train_ohe[np.where(y_train == 0)[0], 1] = 1

    X_init, y_init = compute_pos_neg_sets(X_train,
                                          y_score,
                                          np.array(y_train, dtype=bool),
                                          score_threshold)

    y_init_ohe = np.zeros((len(y_init), 2))
    y_init_ohe[np.where(y_init == 1)[0], 0] = 1
    y_init_ohe[np.where(y_init == 0)[0], 1] = 1

    X_init_cpu = torch.tensor(X_train)
    X_init_cpu = X_init_cpu.float()
    X_val_cpu = torch.tensor(X_val)
    X_val_cpu = X_val_cpu.float()

    max_epochs = 1000
    for i in range(max_epochs):
        # Get the training data generator
        X_gen = DataGenerator(X_init,
                              y_init_ohe,
                              random_state=random_state)

        # Wrap it in a data loader
        X_gen = torch.utils.data.DataLoader(X_gen,
                                            pin_memory=True,
                                            num_workers=2,
                                            batch_size=32,
                                            shuffle=True)
        # Train the model
        sklearn_ap = model.fit_generator(X_gen,
                                         optimizer,
                                         X_valid=X_val,
                                         y_valid=y_val_ohe,
                                         max_epochs=2,
                                         validation_iter=5000000,
                                         device=args.device, 
                                         prev_val_precision=sklearn_ap)

        val_ap.append(sklearn_ap)

        y_score_val = model.predict(X_val_cpu, device=args.device)

        score_threshold, num_pos = compute_fdr_score(y_score_val[:, 0],
                                                     np.array(y_val, dtype=bool),
                                                     fdr_threshold=0.05)
        all_num_pos.append(num_pos)
        
        print(f"Validation CNN epoch {i} average precision: {sklearn_ap}, "
          f" Number of positives at FDR of 5% are: {num_pos}")

        y_score = model.predict(X_init_cpu, device=args.device)

        X_init, y_init = compute_pos_neg_sets(X_train,
                                              y_score[:, 0],
                                              np.array(y_train, dtype=bool),
                                              score_threshold)

        y_init_ohe = np.zeros((len(y_init), 2))
        y_init_ohe[np.where(y_init == 1)[0], 0] = 1
        y_init_ohe[np.where(y_init == 0)[0], 1] = 1

        np.savez(args.save_pos, num_pos=all_num_pos, val_ap=val_ap)
