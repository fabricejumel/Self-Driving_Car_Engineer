**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

#### Data Collection

I tried collecting the data by driving the car in the simulator for both of the tracks by using mouse and keyboard but I didn't succeeded collecting better data compared to the udacity provided. The driving data has driving log that has pointer to the location of the images (frame captured by the simulator) and steering angle, throttle, speed, etc at the time of image captured by the simulator.

#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing "python drive.py model.h5". I have created class geneartes batch of images with different augmentation techiniques. This generator object is passed to train the keras model.

#### Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


#### An appropriate model architecture has been employed and below is an Architecture of the model

The model contain the cropping layer, three convolutional layer the model sucessfully drives the car on track 1. I did not use any dropout.

Layer (type)	Output Shape	Params	Connected to
lambda_1 (Lambda)	(None, 160, 320, 3)	0	input_1[0][0]
cropping2d_1 (Cropping2D)	(None, 60, 300, 3)	0	lambda_1[0][0]
convolution2d_1 (Convolution2D)	(None, 28, 148, 24)	1824	cropping2d_1[0][0]
convolution2d_2 (Convolution2D)	(None, 12, 72, 36)	21636	convolution2d_1[0][0]
convolution2d_3 (Convolution2D)	(None, 4, 34, 48)	43248	convolution2d_2[0][0]
flatten_1 (Flatten)	(None, 6528)	0	convolution2d_3[0][0]
dense_1 (Dense)	(None, 100)	652900	flatten_1[0][0]
dense_2 (Dense)	(None, 50)	5050	dense_1[0][0]
dense_3 (Dense)	(None, 10)	510	dense_2[0][0]
dense_4 (Dense)	(None, 1)	11	dense_3[0][0]

I used the dropout layer to handle the overfitting but I didn't see any changes, so I did not used one.

I used the model based on the Nvidia paper from the behavior cloning videos. I can observe the difference by the normalization in this model, removing the normalization layer resulted in the car becoming unstable on the tracks. I used the three convolution network with depths between 3 to 48. I have used the 5 epoch to train the model. I am assuming the model can run for the 10-15 epoch, after that model will overfit. I used 8 epoch but model started seemes to be overfitting. 

##### Data Augmentation

 - I used the data augmentation by decresing the brightness. And for that I used the function brightness under the DataGenerator class.