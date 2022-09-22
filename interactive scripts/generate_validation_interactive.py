import numpy as np
import argparse


# this function is modified from code by Aric Preive
def parse_arguments():
    """
    Parses the arguments supplied, checks everything so they are in a
    standard format
    """
    # Create custom class for formatting
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, 
            argparse.RawTextHelpFormatter):
        pass

    # Create parser object
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter,
                                     description="A script for flattening matrices for m6A")

    # Split into optional and required args
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')

    # Required arguments
    required.add_argument('-i', '--input_file',
                          help="Path to npz file",
                          required=True)

    # Optional arguments
    optional.add_argument('-t', '--train_file',
                          help="Training output file name",
                          required=False,
                          default=None)
    optional.add_argument('-v', '--valid_file',
                          help="validation output file name",
                          required=False,
                          default=None)
 

    # Append groups together and return
    parser._action_groups.append(optional)
    return parser.parse_args()





def flatten(data_path,
            train_path,
            valid_path):

    data_path_pre = data_path.split(".")[0]

    # name files if not named
    if train_path is None:
        train_path = data_path_pre + "_train.csv"
    if valid_path is None:
        valid_path = data_path_pre  + "_val.csv"



    # load the data
    train_val_data = np.load(data_path, allow_pickle=True)

    # Get the dictionary from the containing relevant data
    train_val_data = train_val_data['save_data_dict'][()]

    # Load training and validation features and labels
    X_train = train_val_data['X_train']
    y_train = train_val_data['y_train']
    X_val = train_val_data['X_val']
    y_val = train_val_data['y_val']




    # Matrix subsetting
    #X_train = X_train[:,0:4, 6:9]
    #X_val = X_val[:,0:4, 6:9]


    # Reshape the data
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2])
    


    # Convert the labels to just 1 dimension
    y_train = y_train[:, 0]
    y_val = y_val[:, 0]
    final_train_data = np.concatenate([y_train[:, np.newaxis], X_train], axis=1)
    final_val_data= np.concatenate([y_val[:, np.newaxis], X_val], axis=1)


    np.savetxt(train_path, final_train_data, delimiter=",")

    np.savetxt(valid_path, final_val_data, delimiter=",")



def main():

    # Parse the arguments supplied
    args = parse_arguments()


    flatten(data_path = args.input_file,
            train_path = args.train_file,
            valid_path = args.valid_file)


if __name__ == "__main__":
    main()
