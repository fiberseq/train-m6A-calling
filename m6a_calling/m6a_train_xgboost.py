import xgboost as xgb
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import gc
import argparse


def train_xgb(train_data_path, val_data_path, out, cv=False, n_train=25_000_000):
    # sm = "PS00075"
    # sm = "PS00109"
    # val_data_path = f"results/{sm}_2/ml/{sm}_2.npz"
    # train_data_path = f"results/{sm}_3/ml/{sm}_3.npz"
    # val data
    val_data = np.load(val_data_path, allow_pickle=True, mmap_mode="r")
    X_val = val_data["features"]
    y_val = val_data["labels"]
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2])
    print(X_val.shape, y_val.shape)
    gc.collect()

    # train data
    train_data = np.load(train_data_path, allow_pickle=True, mmap_mode="r")
    X_train = train_data["features"][0:n_train]
    y_train = train_data["labels"][0:n_train]
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
        help="output model path",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=25_000_000,
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run cross validation, for devs only",
    )
    args = parser.parse_args()
    train_xgb(
        args.train_data_path,
        args.val_data_path,
        args.out,
        cv=args.cv,
        n_train=args.n_train,
    )


if __name__ == "__main__":
    main()
