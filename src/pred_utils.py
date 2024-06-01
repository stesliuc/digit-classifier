from matplotlib import pyplot as plt
import src.backprop_utils as bp
import src.nn_utils as nn


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = bp.forward_prop(W1, b1, W2, b2, X)
    predictions = nn.get_predictions(A2)
    return predictions


def test_prediction(index, X, Y, W1, b1, W2, b2):
    current_image = X[:, index, None]
    prediction = make_predictions(X[:, index, None], W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.show()
