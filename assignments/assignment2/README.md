This assignment covers keypoint matching and image stitching with SIFT and RANSAC.

# Keypoint Detectors

The first part of assignment 2 is all about keypoint detection. Being able to work with and evaluate a range of keypoint detectors is important for many computer vision tasks. To evaluate the efficacy of these detectors, you will evaluate them on the CIFAR-10 dataset.

You will evaluate Histogram of Oriented Gradients and SIFT features both qualitatively and quantitatively. You will then use these features to train a simple classifier and evaluate the performance of each feature set on the CIFAR-10 dataset.

## Keypoint Matching

For qualitative evaluation, you will implement a keypoint matching function that takes in two sets of keypoints and returns a list of matching pairs. You will then use this function to plot the matches between two images.

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

# Image Stitching

In the second part of the assignment, you will implement an image stitching solution which computes a transformation matrix used to warp one image so that it is stitched to another. The first part involves estimating a transformation matrix based on keypoints that are matched. These keypoints will probably contain outliers, so these estimations will then be validated as
part of RANSAC.

## Estimate Affine Matrix

Create a function `compute_affine_transform` which takes in a set of points from the source image and their matching points in the destination image. Using these samples, compute the affine transformation matrix using the normal equations. This function should return a $3 × 3$ matrix.

## Estimate Projective Matrix

Create a function `compute_projective_transform` which takes in a set of points from the source image and their matching points in the destination image. Using these samples, compute the projective transformation matrix using the normal equations. This function should return a $3 × 3$ matrix.

**When using this matrix later on, do not forget to apply a perspective divide!**

## RANSAC

Create a function `ransac` which takes in a set of keypoints in the source image and their potential matches in the destination image. Additionally, it should take in parameters to determine the number of iterations that RANSAC can run, the minimum number of samples to fit a model with, and a threshold boundary. Implement the RANSAC function following the pseudocode on the Wikipedia page (or possibly other sources).

## Testing

Test your implementation on the sample images provided on Canvas to verify your result. Compare using RANSAC with an affine model versus projective model, try each approach using the same set of images. RANSAC will not compute the same results each time. Depending on the selection of keypoints, this may not find a best fit. Create a brief report for this section that shows some of the samples you tried and what the outcome was.

Save your code as `stitch_images.py`.
