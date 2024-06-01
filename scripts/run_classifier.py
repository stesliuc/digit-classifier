import argparse
import numpy as np
import pandas as pd
import src.nn_utils as nn


def main(args):
    # Read in data from in-folder

    # Prepare test and train data
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)  # shuffle before splitting into dev and training sets

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255
    _, m_train = X_train.shape

    # Train Neural Net
    W1, b1, W2, b2 = nn.gradient_descent(X_train, Y_train, 0.10, 500)

    # Make predictions
    dev_predictions = nn.make_predictions(X_dev, W1, b1, W2, b2)
    nn.get_accuracy(dev_predictions, Y_dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a digit")
    parser.add_argument("--in_folder", help="the input folder")
    args = parser.parse_args()

    main(args)
