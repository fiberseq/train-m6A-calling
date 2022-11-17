import xgboost as xgb
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import gc
import argparse
import os
import struct
from ctypes import cdll
from ctypes import c_float, c_uint, c_char_p, c_bool

# function to allow xgboost model to be used in rust with gbdt-rs
# taken from:
# https://github.com/mesalock-linux/gbdt-rs/blob/master/examples/convert_xgboost.py
def convert(input_model, objective, output_file):
    model = xgb.Booster()
    model.load_model(input_model)
    tmp_file = output_file + ".gbdt_rs.mid"
    # extract base score
    try:
        with open(input_model, "rb") as f:
            model_format = struct.unpack("cccc", f.read(4))
            model_format = b"".join(model_format)
            if model_format == b"bs64":
                print("This model type is not supported")
            elif model_format != "binf":
                f.seek(0)
            base_score = struct.unpack("f", f.read(4))[0]
    except Exception as e:
        print("error: ", e)
        return 1

    if os.path.exists(tmp_file):
        print(
            "Intermediate file %s exists. Please remove this file or change your output file path"
            % tmp_file
        )
        return 1

    # dump json
    model.dump_model(tmp_file, dump_format="json")

    # add base score to json file
    try:
        with open(output_file, "w") as f:
            f.write(repr(base_score) + "\n")
            with open(tmp_file) as f2:
                for line in f2.readlines():
                    f.write(line)
    except Exception as e:
        print("error: ", e)
        os.remove(tmp_file)
        return 1

    os.remove(tmp_file)
    return 0


def train_xgb(
    train_data_path, val_data_path, out, cv=False, n_train=25_000_000, n_val=10_000_000
):
    # val data
    val_data = np.load(val_data_path, allow_pickle=True, mmap_mode="r")
    X_val = val_data["features"]
    y_val = val_data["labels"]
    n_val = min(y_val.shape[0], n_val)
    X_val = X_val[0:n_val]
    y_val = y_val[0:n_val]
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2])
    print(X_val.shape, y_val.shape)
    gc.collect()

    # train data
    train_data = np.load(train_data_path, allow_pickle=True, mmap_mode="r")
    X_train = train_data["features"]
    y_train = train_data["labels"]
    n_train = min(y_train.shape[0], n_train)
    X_train = X_train[0:n_train]
    y_train = y_train[0:n_train]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    print(X_train.shape, y_train.shape)
    gc.collect()

    # ML
    print("Starting ML")

    # weight for training based on uneven label data
    scale_pos_weight = np.sum(y_val == 0) * 1.0 / np.sum(y_val == 1)

    if cv:
        estimator = XGBClassifier(
            objective="binary:logistic", seed=42, eval_metric="aucpr", nthread=20
        )

        parameters = {
            "n_estimators": [100, 150],
            "scale_pos_weight": [scale_pos_weight],
            "max_depth": [6, 8],
            "min_child_weight": [100, 300],
            "gamma": [1, 10],
        }

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=parameters,
            scoring="average_precision",  # "roc_auc",
            n_jobs=2,
            cv=2,
            verbose=4,
        )

        # Train model
        grid_search.fit(X_val, y_val)

        bst = grid_search.best_estimator_
        print(type(bst))

        # Save model
        bst.save_model(f"models/xgboost_CV_0.81.bin")

        # results show that these parameters make the most sense
        # {'gamma': 1, 'max_depth': 8, 'min_child_weight': 100, 'n_estimators': 150, 'scale_pos_weight': 21.852819627916418}

    else:
        dval = xgb.DMatrix(X_val, label=y_val)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        print("xgb matrix made")

        gamma = 1
        max_depth = 8  # 6
        min_child_weight = 100
        n_estimators = 150  # 100
        param = {
            "gamma": gamma,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "n_estimators": n_estimators,
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": ["aucpr", "auc"],
        }
        num_round = 50
        eval_s = [(dtrain, "train"), (dval, "validation")]

        # Train model
        bst = xgb.train(param, dtrain, num_round, evals=eval_s)

        # Save model
        bst.save_model(out)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "train_data_path",
        help="Path to training data, (.npz)",
    )
    parser.add_argument(
        "val_data_path",
        help="Path to validation data, (.npz)",
    )
    parser.add_argument(
        "out",
        help="output model path, should have a .bin extension",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=100_000_000,
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=10_000_000,
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run cross validation, for devs only",
    )
    parser.add_argument("--objective", default="binary:logistic")
    args = parser.parse_args()
    train_xgb(
        args.train_data_path,
        args.val_data_path,
        args.out,
        cv=args.cv,
        n_train=args.n_train,
        n_val=args.n_val,
    )
    convert(args.out, args.objective, f"{args.out}.json")


if __name__ == "__main__":
    main()
