import numpy as np

# TODO: Create feature processing functions for SIFT and HOG

if __name__ == "__main__":
    # Load the pre-split data
    data = np.load("cifar10.npz", allow_pickle=True)

    # TODO: Extract features from the training data
    X_train = data["X_train"].astype(np.uint8)
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(X_train[0])

    # TODO: Extract features from the testing data

    # TODO: Save the extracted features to a file
