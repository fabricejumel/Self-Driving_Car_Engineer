##Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

- Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I used pretty much the same approach taught by the tutor in tutorial. I started by reading in all the vehicle and non-vehicle images. I randomly selected 5 images from each category which are shown in output folder. I define all the functions to assist with extracting features using HoG, Spatial Binning, and Color Histogram. The  function iterates over all images passed into the next function and extracts features by calling the other functions. The function is called for the list of images, which contain a car and the non-car. The results of these extracted features for each set of images are then stacked and normalized and then split into training and testing datasets.

####2. Explain how you settled on your final choice of HOG parameters.

I iterated over several colorspaces without much changes to the different parameters required for skimage's hog method. I based my selection eventually on the test set accuracy I achieved after training my classifier. Here are some parameters.
cspace = 'HSV', spatial_size = (16, 16), hist_bins = 32, hist_range = (0, 256), orient = 9, pix_per_cell = 8, cell_per_block = 2, hog_channel = "ALL"

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Once I obtained my training and test set from extracted features, I utilized sklearn's Linear Support Vector Classification to train a classifier. I obtained a test accuracy of 99.35%. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

The window size is 0.75 overlap for all of them. The tradeoff here was in terms of processing speed. I noticed that I could perhaps get better results with smaller window sizes and higher overlap, but this could result to run video much slower.

####2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to try to minimize false positives and reliably detect cars?

I run my pipeline for individual test images. I then call the search windows function. This function identifies windows in the images where the classifier detects a vehicle.

###Video Implementation

Output Image folder contain, randomely selected five vehicle and non-vehicle images. From the test images, the selected region where the car object find from the image. And I am also attaching the final video output where the region is selected from the frames.

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

The results was not accurate, when there are two cars comes to a picture, the frames seems to be distructed, and after a second it again starts recovering.
