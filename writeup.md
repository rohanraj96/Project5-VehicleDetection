---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/threshold.png
[image3]: ./output_images/derivation.png
[image4]: ./output_images/windows.png
[image5]: ./output_images/heatmap.png
[image6]: ./output_images/labels.png
[image7]: ./output_images/output_bbox.png
[image8]: ./output_images/choose.png
[video1]: ./2018/submit.mp4

### SUMMARY OF FILES AND FOLDERS THAT I CREATED

#### 1. /2018

Contains project video (submit.mp4) and final code that was used for submission video (submit.ipynb). Also has a test.ipynb where I used a class implementation again with some changes but it didn't work so well. The class is defined in detector.py.

#### 2. /models

Contains various models and their normalized scalers when I was experimenting with hog parameters et al.

#### 3. /output_images

Images to be used in this writeup

#### 4. /videos

Miscellaneous videos that were rendered for various experiments

#### 5. features.py

Python script that has all functions related to feature extraction which was used in a previous approach that I was working on.

#### 6. run.py

Class implementation to record history and average out frames.

#### 7. sliding_window.py

Functions related to sliding window technique.

#### 8. train.ipynb and test.ipynb

Very clean approach to solving this problem using 2 seperate train and test notebooks which also used a class. Unfortunately, this didn't work so well.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the eighth code cell of the IPython notebook train.ipynb.  

I started by reading in all the `vehicle` and `non-vehicle` images. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters as shown in the following figure:

![alt text][image8]

Of which YCrCb with ALL channels gave the best results. I also experimented with various orientations, pix_per_cell etc values.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using C=0.1 after doing a grid search which gave best performace for C = 0.1. This can be seen in cell 4 of 2018/2018.ipynb.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to calculate features for entire image at once and then sub-sample as discussed in the lecture and applied a sliding window search of 50% square overlap over a cropped region of every frame, where we can expect to find a car. This is a kind of hard-coding and is a major drawback of my pipeline but it works well on the project video. I tried various overlap percentages and chose a window size of (64,64) for different scales of the image. This works similar to using different window sizes.

Here is how the detections look:

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

For starters I decided to come up with a formula that would give me scaled windows for each given window area. The derivation looked something like this: 

![alt text][image3]

This worked poorly, so I tried using global variables to average out the frames but it wasn't very flexible and the code was very messy. 
After this, I used a second approach using a class to store the history of past 'n' frames, where n is a tunable parameter, to get more smooth results but the results were not good as seen in 'videos/with_class.mp4'. Ultimately I did detection using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector using a find_cars() class which gives a nice result.  Here are some example images:

![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./2018/submit.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and kept adding the heatmaps to a queue of size 5 and only predicted a car if the superimposed heatmaps of all the combined 5 images were above a defined threshold. I also cropped some area from the x axis to minimise false positives. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in this cummulative heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is few frames and its corresponding heatmap:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One major flaw with my pipeline is discarding half of the frame to remove false positives even though there is a possibility of a car right in front of us. This can be definitely improved by training a better classifier using hard negative mining or using a better classifier.
My detections are also very shaky, which can be improved by averaging previous frames to give a smooth output.
Another drawback is that the classifier is too slow to work in real-time as shown in train.ipynb.
