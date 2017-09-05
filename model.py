import pickle
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
from helper_functions import *

"""
Here we load a dataset of labeled vehicle/not-vehicle images, extract relevant features, normalize them, 
and finally choose and train a classifier (SVM) to make predictions in new images (video frames).
"""

t0 = time.time()
color_space = 'YCrCb'       # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9                  # number of HOG orientations
pix_per_cell = 8            # HOG pixels per cell
cell_per_block = 2          # HOG cells per block
hog_channel = 'ALL'         # HOG features, Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)     # Spatial binning dimensions doing this in $color_space
hist_bins = 16              # Number of histogram bins, doing this in $color_space
spatial_feat = True         # Spatial features on or off
hist_feat = True            # Histogram features on or off
hog_feat = True             # HOG features on or off


# Divide up into cars and notcars from the combined dataset (GTI, KITTI, Video)
images = glob.glob('../projects_data/p5-data/dataset/*/*/*.png')
cars = []
notcars = []
for image in images:
    if 'non-vehicles' in image:
        notcars.append(image)
    else:
        cars.append(image)


# Extract features from the two classes: I'm including HOG, color histogram and spatial features
car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)


# Since the different nature of the features we need to normalize the features vector for our classifier
X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)      # Fit a per-column scaler
scaled_X = X_scaler.transform(X)        # Apply the scaler to X


# Now I need to divide the data between training and test sets
# First I define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
# Split up data into randomized training and test in a proportion of 80% to 20%
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
print('Feature vector length:', len(X_train[0]))


# Finally we choose and train a classifier. In this case we'll use a support vector machine classifier
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t1 = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t1, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


# We export data we'll use in the following scripts using pickle.
# Model and feature extraction parameters:
with open('pickle_files/model_and_pars.pickle', 'wb') as f:
    pickle.dump([svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size,
                 hist_bins, spatial_feat, hist_feat, hog_feat], f)
# List of names of cars and not-cars in the dataset:
with open('pickle_files/cars_not_cars.pickle', 'wb') as f:
    pickle.dump([cars, notcars], f)
# Check the total execution time:
tf = time.time()
print(round(tf - t0, 2), 'Total Time')
