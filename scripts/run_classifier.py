import argparse
import numpy as np
import pandas as pd
import os
import src.nn_utils as nn
import src.pred_utils as pred


def main(args):
    # Read in data from in-folder
    data = pd.read_csv(os.path.join(args.data_folder, "train.csv"))

    # Prepare test and train data
    data = np.array(data)
    m, n = data.shape

    # Shuffle data before splitting into test and train sets
    np.random.shuffle(data)

    data_test = data[0:1000].T
    Y_test = data_test[0]
    X_test = data_test[1:n]
    X_test = X_test / 255

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255

    _, m_train = X_train.shape

    # Train Neural Net
    W1, b1, W2, b2 = nn.gradient_descent(X_train, Y_train, m, 0.10, 500)

    # Make predictions
    test_preds = pred.make_predictions(X_test, W1, b1, W2, b2)
    accuracy = nn.get_accuracy(test_preds, Y_test)
    print(accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a digit")
    parser.add_argument("--data_folder", help="the input folder")
    parser.add_argument(
        "--data_type",
        choices=["train", "test"],
        default="train",
        help="train or test dataaset",
    )
    args = parser.parse_args()

    main(args)
