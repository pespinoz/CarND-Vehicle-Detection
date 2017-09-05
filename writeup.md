
## **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[im01]: ./output_images/car_not_car.jpg
[im02]: ./output_images/hog_features_car.jpg
[im03]: ./output_images/color_spatial_feats_car.jpg
[im04]: ./output_images/hog_features_nocar.jpg
[im05]: ./output_images/color_spatial_feats_nocar.jpg
[im06]: ./output_images/windows_on_testims.jpg
[im07]: ./output_images/990.jpg
[im08]: ./output_images/991.jpg
[im09]: ./output_images/992.jpg
[im10]: ./output_images/993.jpg
[im11]: ./output_images/994.jpg
[im12]: ./output_images/995.jpg
[im13]: ./output_images/996.jpg
[im14]: ./output_images/final.jpg
[vi01]: ./project_video_output.mp4

---

### Files Included:
My project includes the following files in the top level directory, "./":
* `helper_functions.py:` This file contains a several functions that aid on the implementation of the video pipeline (mostly taken from the Udacity lectures).  
* `model.py:` Loads a dataset of labeled vehicle/not-vehicle images, extract relevant features, normalize them, 
and finally choose and train a classifier (SVM).
* `detect_and_filter.py:` Computes bounding boxes for detected vehicles and filter false positives in a time-series set of images (extracted from the project video using `ffmpeg`).
* `pipeline.py:` This file contains the implementation of the video pipeline.
* `plots.py:` Outputs the plots used in this writeup file.
* `writeup.md:` You're reading it!
* `project_video_output.mp4:` A video successfully showing the vehicle detection pipeline.

It also includes the following folders:
* `output_images/`: Here I store the images produced in this project.
* `pickle_files/`: Files here store the trained model, feature extraction parameters, sliding window parameters, and false-positives filtering parameters. These are all used to execute `pipeline.py`, the core of this project.

---

### Running the Code:

The video pipeline is implemented in `./pipeline.py`. This generates our [output mp4 file][vi01]. All the functions used in this pipeline are implemented in the file `helper_functions.py`. 

Additionally, we produce the plots illustrating this file in `./plots.py`, and `./detect_and_filter.py`. 

---


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines #14-50 of the file `model.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images (64x64) from the dataset. This one is composed by GTI and KITTI images, as well as frames extracted from the project video itself.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][im01]

After reading them in I saved the dataset file names in a pickle file called `cars_not_cars.pickle`. I'll be using these variables later for plotting purposes. 

The next step is to extract features from the images. Here I have several options, as I can consider i) histogram of oriented gradients (HOGs), ii) color histograms, and iii) raw pixel values that have been spatially binned down from the image. We can consider all of these options, either separately or combined to form a feature vector we can feed the classifier.
 
 For the HOG feature extraction I tried different combinations of parameters in the `hog()` function, settling down for the following ones:

| HOG Parameters     | Values        | 
|:------------------:|:-------------:| 
| color space        | YCrCb         | 
| channels           | all           |
| orientations       | 9             |
| pixels per cell    | 8             |
| cells per block    | 2             |
 
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Here is an example for the `car` class using the `YCrCb` color space and HOG parameters specified in the Table above:

![alt text][im02]

Additionally I chose to combine all features that could be extracted. Therefore I did consider color histograms (16 bins) and raw pixel values (spatially binned down to 16x16 images). An example of these features for the same image shown above is illustrated next:

![alt text][im03]

Similarly, I performed the same feature extraction process for the `not-car` class. In the next plot we show an example of a dataset image and its HOG visualization with the above specified HOG parameters:

![alt text][im04]

And the same for color histograms (16 bins) and raw pixel values (spatially binned down to 16x16 images). Note that any of the features we chose look significantly different between the `car` and `not-car` classes, which is our objective to train the classifier efficiently. 

![alt text][im05]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters for the HOG visualization, starting with the ones given in the Udacity lectures and then departing from those slightly. However, after visual inspection, I found that the shape differences between the classes were maximized with `orientations=9`, `pixels per cell=8`, and `cells per block=2`. It is worth noticing that all channels of the [YCbCr](https://en.wikipedia.org/wiki/YCbCr) colorspace were considered to derive the HOG features.

This choice of parameters also makes the length of the vector of extracted features manageable for my machine. Finally the parameter selection, not only of HOG but of `hist_bins` and `spatial_size` too, allows for a high score in the testing of the classifier. We will analize this further in the following Section. 
     
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the extracted features from HOG, color histograms, and raw pixel values in lines #54-73 of `model.py`. As mentioned above, the parameter selection was: `orientations=9`, `pixels per cell=8`, and `cells per block=2` for HOG, `hist_bins=16` for color histogram, and `spatial_size=(16, 16)` for the subsampling of the dataset images. Concatenating all these features into a single vector gave us a vector length of 6108.

The next step is to normalize the data so all the type of features contribute equally during the training stage. I did this with sklearn's `StandardScaler()` function (lines #55-56 of `model.py`). The training-test split was the usual 80%-20%, with sample randomnization. It was achieved with the `train_test_split()` function in line #64.

Finally, we set a linear SVM classifier in lines #70-73. After training, the test score reaches 99.1%, which validates our feature extraction strategy described in the previous Section.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

A sliding window search is performed in the image, then our classifier predicts if there is a car or not in each window. 

However, it's computationally expensive to make a prediction in each window. One way to alleviate this is to avoid zones of the image we know cars can't be found. For example: the sky or where the trees are.

I introduced a new function to perform the window search in the image, called `slide_variable_window()` in `helper_functions.py` line #131. The advantage with respect to the Udacity implementation is that my function delivers multi-scale windows. This is appropriate as other vehicles look smaller the farther away they are down the road. In other words, the apparent size of another vehicle decreases with decreasing image y-coordinate. `slide_variable_window()` begins with a 64 pixel window size to detect vehicles far away, and increases to 132 pixel windows for nearby vehicle detection. These sizes, as well as the overlaps both in the x- and y-coordinates, were determined by trial and error such that detection in the test images and the video frames was optimized. The parameters of the sliding window can be found in lines #21-26 of `detect_and_filter.py`, or the Table below:

| Sliding Window Parameters     | Values        | 
|:------------------:|:-------------:    | 
| x-pixel start       | 200              | 
| y-pixel start       | 375              |
| x/y pixel end       | None             |
| x-overlap           | 0.9              |
| y-overlap           | 0.7              |
| xy window size      | 64x64 to 132x132 | 
 
Note that the overlap in the y-coordinate can be relaxed (with respect to its x-counterpart) because we're using multi-scale windows in that axis. Results of this strategy will be presented next.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The multi-scale sliding window search and classifier prediction are implemented in the functions `slide_variable_window()` and `search_windows()` (in `helper_functions.py` lines #131 and 213 respectively). These are the core functions for vehicle detection.
 
Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned raw pixel values and histograms of color in the feature vector, which provided a nice result. Here are some examples from the set of test images included in the project (`plots.py` lines #90-124):

![alt text][im06]

Note that in these images it's common to see overlapping detections that exist for each of the two vehicles. In two of the frames (`test6.jpg` and `test5.jpg`) I find a false positive detections, and in `test3.jpg` we have a non-detection. All these issues can be dealt with in a time-series set of frames, such as the video of our project.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result][vi01]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.
 
 Given the previous history of detection in each frame, I implement a filter that applies a threshold in the cumulative heatmap in order to combine overlapping detections and remove false positives. This is done through a `Tracker` class in `pipeline.py`, lines #26-27. The heatmap is integrated over `n_frames=35` frames (line #42) such that multiple detections get "hot", while transient false positives stay "cool". Then we simply apply a threshold, set to `heat_thres=28` in line #42.  

Here's an example result showing a series of seven frames of video, and their corresponding heatmaps:

![alt text][im07]
![alt text][im08]
![alt text][im09]
![alt text][im10]
![alt text][im11]
![alt text][im12]
![alt text][im13]

Now we identify individual blobs with the `label()` function. These blobs represent individual vehicles (see left panel of the next Figure). The resulting bounding boxes drawn onto the last frame in the series (the seventh) are shown in the right panel of the Figure. This is the output of the pipeline from which we build the [output video][vi01] of this Project.

![alt text][im14]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach was to start by trying several feature-extraction and detection approaches with the set of test images included in the project repository. This stage was intensive regarding the fine-tuning of many parameters. Next, I applied the obtained knowledge to the video pipeline.

When I implemented the video pipeline, I also added a `Tracker` class that basically combines overlapping detections and applies a threshold in the cumulative heatmap to remove false positives. This strategy performs well in the project video, but it's evident that the fine-tuned parameters used here are not robust. 

Many of the parameters become obsolete when environmental and driving conditions change. For example, our hard coded search-region would not be useful if driving in a road with steep slopes. Our feature extraction (HOG and color histograms) procedure would need to change if brightness or weather conditions vary. And more noticeably, the parameters that control the overlapping detections and false positives removal (`n_frames` and `heat_thres`) depend entirely on the relative speed of the cars on the road.

In that sense it is possible that deep-learning approaches (SSD, YOLO) can derive a generalized, robust model. These strategies are also capable of processing several frames per second (~30 fps). This makes an on-line implementation feasible, unlike our present pipeline which is quite slow.
