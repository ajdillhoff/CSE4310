This assignment covers keypoint matching and image stitching with SIFT and RANSAC.

# Keypoint Detectors

The first part of assignment 2 is all about keypoint detection. Being able to work with and evaluate a range of keypoint detectors is important for many computer vision tasks. To evaluate the efficacy of these detectors, you will evaluate them on the CIFAR-10 dataset.

You will evaluate Histogram of Oriented Gradients and SIFT features both qualitatively and quantitatively. You are then tasked to use these features to train a simple classifier and evaluate the performance of each feature set on the CIFAR-10 dataset.

## Keypoint Matching of SIFT Features

For qualitative evaluation, implement a keypoint matching function that takes in two sets of SIFT keypoints and returns a list of matching pairs. You will then use this function to plot the matches between two images.

Create a matching function that, given two sets of keypoint features, returns a list of indices of matching pairs. That is, pair $(i, j)$ in the list indicates a match between the feature at index $i$ in the source image with the feature at index $j$ in the second image.

**You should write the function yourself, but are encouraged to also verify your function with `match_descriptors` from `skimage.feature`.**

### Plot Keypoint Matches

Create a plotting function that combines two input images of the same size side-by-side and plots the keypoints detected in each image. Additionally, plot lines between the keypoints showing which ones have been matched together. An example of this function is provided in `ransac.ipynb` in this same repository.

## Evaluating Keypoint Detectors

To test the efficacy of both HOG and SIFT, you will train two simple classifiers on the CIFAR-10 dataset, one with HOG features and one with SIFT features. You will then evaluate the performance of these classifiers on the test set.

### Bag of Visual Words

To train a classifier using SIFT or HOG features, you will first need to create a bag of visual words. This is a set of representative features that will be used to train the classifier. You will then use the bag of visual words to create a histogram of visual words for each image in the dataset. This histogram will be used as the feature vector for each image. Feel free to use the demo code that was presented in class to help you with this part of the assignment.

**Before moving to the next part, make sure you have run `load_and_split.py` to download the CIFAR-10 dataset and split it into training and testing sets.**

In a Python script named `feature_extraction.py`, implement functions to create a bag of visual words using both SIFT and HOG features. In the `main` part of the file, take the data that is loaded by `numpy` and process the training and testing data. After converting them to the fixed length feature vectors, save them to a file. Each file should store a dictionary object containing `X_train`, `y_train`, `X_test`, and `y_test`. You will use these files in the next part of the assignment. You should have two files, one for HOG and one for SIFT.

### Classification with Support Vector Machines

For classification, create two files `evaluate_hog.py` and `evaluate_sift.py` that train a support vector machine (SVM) using the HOG and SIFT features, respectively. You can use the `sklearn` library to train the SVM as shown in the example code. You should use the default parameters for the SVM. 

Once they are trained, evaluate the performance of the classifiers on the test set. You should report the accuracy of the classifiers on the test set.

## Requirements

Your final implementation should include any necessary comments and be easy to follow and interpret. You can use either a notebook or a Python script. Whatever you choose, there should be a separate file for each feature detector. In a separate document, write a brief summary that includes:

1. the number of features extracted using each approach
2. the number of correct matches found
3. the accuracy of the classifiers when evaluated on the test set

# Questions

1. Describe a process for performing keypoint matching using HOG features. The challenge here is that HOG features are typically generated for the entire image.
2. Give your own interpretation of the results. Why do you think one feature set performed better than the other? Consider the efficiency of the feature extraction process and the quality of the features themselves.