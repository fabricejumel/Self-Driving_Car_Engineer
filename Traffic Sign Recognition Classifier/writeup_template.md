#**Traffic Sign Recognition** 

##Writeup Template

---

**Steps took to build the project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

**External library usage

cv2, tensorflow, numpy, pickle, random, collection, os, os.path, heapq


## Rubric Points

###Point that we can take to make the model run accurate

There are many method to generate existing data with different ways.
####1. Augmentation : 

    The amount of data we have is not sufficient for a model to generalise well. It is also fairly unbalanced, and some classes are represented to significantly lower extent than the others. But we will fix this with data augmentation!

####Methods in augmentation

1. Flipping
2. Rotation and Projection

Once we will generateadditional data with methods like augmentation, we will be having lot more data to train. In this way model/architeture will be train better in result, architecture will perform bettter on test model/real world example.

####The submission provides details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

Convolutional 16x16x18 -> MaxPool -> ReLu -> Convolutional 8x8x36 -> MaxPool -> ReLu -> Flatten (2304) -> Fully Connected (128 units) -> Fully Connected (43 units, output)

####The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

Learning rate: 0.003
Batch size: 200
Epochs: 15000
Hyperparameters: Dropout keep probability for training = 1, standard deviation= 0.05

####The submission describes the approach to finding a solution. 

I started off by using the LeNet network architecutre. Which include the Convolotuon layer, Fully connected layer and Flatten layer which include the relu, dropout, softmax function to improve the accuracy of the model.
The model's final test accuracy was around 0.93 and run the model for the for 15000 epoch and top accuracy found on 12200.

####The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to any particular qualities of the images or traffic signs in the images that may be of interest, such as whether they would be difficult for the model to classify. 

The five images I have added are: 1) 60 km/h sign, 2) Slippery road sign, 3) Yield sign, 4) Ahead only sign and 5) No entry sign.
This model might face difficulty to find images with low resolution(this images are high resolution) or extremely blur condition, the sign with more than one symbol and the sign in heavy snow condition where snow is stick to the sign where only part of the sign is visible. And to test the images with others where the model gives 95% accuracy for amongst all the images.

####The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set. 

The accuracy of the Images form the five Images approximate 95%.

####The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions. -- remianing

For the "Speed limit (60km/h)" Sign: The model was 97% certain about the prediction, and the prediction was correct
For the "Slippery road" Sign: The model was 90% certain about the prediction, and the prediction was correct
For the "Ahead only" Sign: The model was 85% certain about the prediction, eventhough it was correct
For the "No entry" Sign: The model was 90% certain about the prediction, and the prediction was correct
For the "Yield" Sign: The model was 100% certain about the prediction, and the prediction was correct