#from sklearn.metrics import precision_score, recall_score, accuracy_score
import xgboost as xgb
import numpy as np
import argparse
import sklearn.metrics as metrics


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
                                     description="A script for running XGboost on m6A data")

    # Split into optional and required args
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')

    # Required arguments
    required.add_argument('-t', '--train_file',
                          help="Training input file name",
                          required=True)

    required.add_argument('-v', '--valid_file',
                          help="Validation input file name",
                          required=True)

    required.add_argument('-m', '--model_name',
                          help="model name for saving",
                          required=True)
    

    

    # Optional arguments
    optional.add_argument('-n', '--n_rounds_total',
                          help="Total number of rounds to run",
                          required=False,
                          default=100)

    optional.add_argument('-i', '--n_rounds_start',
                          help="Number of rounds to run before comparing models",
                          required=False,
                          default=1)

    optional.add_argument('-d', '--depth',
                          help="depth for XgBoost",
                          required=False,
                          default=6)

    optional.add_argument('-p', '--patience',
                          help="number of iterations with falling AUC before stopped",
                          required=False,
                          default=5)
 

    # Append groups together and return
    parser._action_groups.append(optional)
    return parser.parse_args()



def run_XG(train_file,
           validation_file,
           num_round,
           num_round_start,
           model_name,
           depth,
           pat):


    # intitalize patience counter
    p = 0
    

    # read in data
    dtrain = xgb.DMatrix(train_file + '?format=csv&label_column=0')
    dtest = xgb.DMatrix(validation_file + '?format=csv&label_column=0')


    # specify parameters via map
    param = {'max_depth':depth, 'objective':'binary:logistic', 'eval_metric':'auc'}
    eval_s=[(dtrain, "train"), (dtest, "test")]

    # intialize model
    bst = xgb.train(param, dtrain, num_round_start, evals=eval_s)

    # "intialize" best model
    bst.save_model('xgboost.{}_best.json'.format(model_name))
    # "intialize" current model
    bst.save_model('xgboost.current.json')

    # save AUC
    preds = bst.predict(dtest)
    fpr, tpr, thresh = metrics.roc_curve(dtest.get_label(), preds, pos_label=1)
    auc_best = metrics.auc(fpr, tpr)


    # we already did some rounds so update num_round
    num_round = num_round - num_round_start


    # update the model
    for i in range(num_round):
        
        # train for 1 iteration starting with last model
        bst = xgb.train(param, dtrain, 1, evals=eval_s, xgb_model='xgboost.current.json')
        
        # update current model
        bst.save_model('xgboost.current.json')

        
        # calc AUC
        preds = bst.predict(dtest)
        fpr, tpr, thresh = metrics.roc_curve(dtest.get_label(), preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        # If AUC is still getting better we want to save the model as current best
        if auc > auc_best:

            # update auc_best
            auc_best = auc

            # save the model
            bst.save_model('xgboost.{}_best.json'.format(model_name))

            # reset patience
            p = 0

        # AUC is getting worse so we need to update patience and stop if we are out of patience
        else:
            p += 1

            if p >= pat:
                break
        





def main():

    # Parse the arguments supplied
    args = parse_arguments()

    # run the XG boost
    run_XG(train_file = args.train_file,
           validation_file = args.valid_file,
           num_round = int(args.n_rounds_total),
           num_round_start = int(args.n_rounds_start),
           model_name = args.model_name,
           depth = int(args.depth),
           pat = int(args.patience))


if __name__ == "__main__":
    main()
