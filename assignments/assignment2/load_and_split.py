# Load and split the data into training and test sets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import numpy as np


if __name__ == "__main__":
    # Load data
    cifar10 = fetch_openml('CIFAR_10_small', version=1, cache=True)
    X = cifar10.data
    y = cifar10.target

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Combine data into a dictionary
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    # Save the dictionary to a file
    np.savez("cifar10.npz", **data)