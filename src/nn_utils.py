import numpy as np
import src.backprop_utils as bp


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, m, alpha, iterations):
    W1, b1, W2, b2 = bp.init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = bp.forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = bp.backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m)
        W1, b1, W2, b2 = bp.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
