# udacity-carnd-behavioral-cloning-p3
Udacity's Car Nano degree - Behavioral Cloning Project



The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sample_images/centerPlot.png "Center Plot"
[image2]: ./sample_images/leftPlot.png "Left Plot"
[image3]: ./sample_images/rightPlot.png "Right Plot"
[image4]: ./sample_images/image2.png "Color Channel image"
[image5]: ./sample_images/image1.png "Processed Image"

### A basic retrospective
This was one of the toughest and the most interesting projects I have done so far - and I learned it the hard way that excessive data doesn't solve all the problems. It was quite clear over the course of project how bad data can stupendously worsen the results. Some major takeaways for this project for me were:
1. Garbage in Garbage out holds very very true for deep learning
2. Sensible randomization while training the model will go a long way
3. Some insight into how things would work in practice.

### Project files:
Contains following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode.
* model.h5 containing a trained convolution neural network
* video.mp4 created using video.py shows recording of driving around a lap and a bit more. The video seems to have been created using the front camera.
* README.md which follows the writeup_template summarizing the results

Note: Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

### Code structure
* First section that just takes care of all imports
* Second section defines utility methods for data load, augmentation and batch generation
* Third and last section contains the code for training and saving the convolution neural network.

### Model Architecture and approach
1. I decided to start with slightly modified LeNet architecture which we learned and used in the traffic-sign-classification project. I was familiar with it and I thought this would be a good start. Didn't work super well and my car kept leaving the road quite early on - It was a quick realization to use a more powerful architecture - Enter NVIDIA
2. From the lectures, I Remembered NVIDIA architecture and VGG. Just really as a toss-up I decided to start with NVIDIAs architecture and it performed well on the first run (failed close to the bridge) and I pretty much stuck with it for the rest of the project. You will be able to see this in `model.py`. The only change is my own zero-mean-centering layer and dropouts after the large Fully Connected Layers.
3. My model consists of:
* Lambda layer for zero-mean-centering
* Cropping layer to get only relevant information out
* Conv layer(24x5x5) with stride 2 and Relu activation
* Conv layer(36x5x5) with stride 2 and Relu activation
* Conv layer(48x5x5) with stride 2 and Relu activation
* Conv layer(64x3x3) with and Relu activation
* Flatten layer
* 3 FC layers with 50% dropouts (I really wanted to avoid overfitting from the get-go hence kept this at 50%)
* The model used an adam optimizer, so the learning rate was not tuned manually


### Data load and exploration
This is where I spent most of my time for this project. I cannot re-iterate that statement more. This was the most challenging, frustrating yet interesting part of this project. Here are some steps
1. I first started working with the sample Udacity data. Some things could be noticed quite quickly on exploring the data using histograms. The data was quite heavily distributed close to zero angles and smaller angles. With a basic 5-epoch training on an NVIDIA model it went off road fairly often at different parts of the track (most noticeably the bridge)
2. At this point I went down a rabbit hole of collecting data. I started first with just gathering some bridge data and some curve turns which made no difference to my results (since it was probably not quality data), but before I knew I had collected 2 laps of driving in the opposite direction, plenty of curve fixes and 2 laps on the second track. Also, the interestingly, every time the model failed - I just attributed it to less data and kept collecting more or replacing a current set with another one.
3. Initially I decided to flip every image and brighten all of them that had an angle value larger than zero to try to create some balance and before I knew I had ~138K samples. It took every training iteration really long to run and still didn't get me the right results (purely because of bad data quality and not enough randomization to avoid overfitting). This was quite easily seen by the lower training loss and a varying validation loss.
4. Had to learn the hard way that proper data augmentation and randomization will help me solve some of my issues. Thanks to some mentors on the forums who helped out.
5. Finally added 2 really specific sets for side-to-center recovery and curve driving to Udacity's sample set.

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Data Augmentation and Training strategy
After some suggestions from mentors on the forums. I took the augmentation and randomization approach. My each generator call does this:
0. Randomly shuffles the data.
1. randomly selecting from right, left and center cameras
2. randomly flipping 70% of the samples
3. randomly applying brightening and XY shifts to images
4. randomly dropping zero angles.
5. Applying a smaller correction to right/left cameras when angle was zero and higher when angle was non-zero. This could be improved with a better function - but this works well for now.
I also split my data into a 70-30 training-validation split.

It took me quite a bit of time to play around with these various ideas and I went through multiple permutations to get the right combination and better accuracy.

![alt text][image3]
![alt text][image4]
