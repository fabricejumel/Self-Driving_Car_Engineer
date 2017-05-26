## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply various masks different color spaces over the image to try to isolate the points of the lane markers.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Some of the description about the model.

I used a clustering algorithm to try to provide a rough classification of the lanes in each image. This is done by slicing the binary image which identifies the lane markes. I used the normalization function to normalize the image. 

Some of the description about the pipeline.

After performing calibration matrix on images, I normalize the Images, then apply the various mask over the images like sobel. The function used to generate a green area.

I am attaching the undistorted images into the output folder where model will run on all the images as shown in tutorial. Then for the same number of input images I am finding the the grid of the images which again I am attching to the output folder. Then I am displaying the birdeye image for couple of Image. Then I am running the pipeline model to the image and then useing to same pipeline for the video to produce the results.

for the pipeline output to the image, the highlighted area comes as green highlighted color and the video also highlighted as the same color.
