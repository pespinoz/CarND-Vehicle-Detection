import matplotlib.pyplot as plt
import pickle
from helper_functions import *
import glob

"""
This file loads pickle files from model.py and detect_and_filter.py to generate the plots needed in the project.
These plots include i) Car/Not-Car, ii) Car extracted features, iii) Not-car extracted features, and iv) bounding
boxes annotated in the set of test images included in the project 
"""

with open('pickle_files/cars_not_cars.pickle', 'rb') as f:
    cars, notcars = pickle.load(f)

with open('pickle_files/model_and_pars.pickle', 'rb') as f:
    svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, \
    spatial_feat, hist_feat, hog_feat = pickle.load(f)

with open('pickle_files/window_search_pars.pickle', 'rb') as f:
    x_start, y_start, x_overlap, y_overlap, xy_window = pickle.load(f)

#######################
# I. Car, Not Car plots

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.imshow(mpimg.imread(cars[np.random.randint(0, len(cars))]))
ax1.set_title('Example of Car Image', fontsize=15)
ax2.imshow(mpimg.imread(notcars[np.random.randint(0, len(notcars))]))
ax2.set_title('Example of Not-Car Image', fontsize=15)
fig.savefig('output_images/car_not_car.jpg', dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
            papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)

##################
# II. Car Features

im_car = mpimg.imread(cars[np.random.randint(0, len(cars))])
gray_car = cv2.cvtColor(im_car, cv2.COLOR_RGB2GRAY)
_, hog_im_car = get_hog_features(gray_car, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
ch1hist_car, _, _, bincen_car, _ = color_hist(cv2.cvtColor(im_car, cv2.COLOR_RGB2YCrCb),
                                              nbins=hist_bins, bins_range=(0, 1))
spatial_car = bin_spatial(cv2.cvtColor(im_car, cv2.COLOR_RGB2YCrCb), size=spatial_size)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.imshow(im_car, cmap='gray')
ax1.set_title('Example of Car Image', fontsize=15)
ax2.imshow(hog_im_car, cmap='gray')
ax2.set_title('Hog Visualization', fontsize=15)
fig.savefig('output_images/hog_features_car.jpg', dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
            papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)

fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
ax3.bar(bincen_car, ch1hist_car[0], width=0.75 * (bincen_car[1] - bincen_car[0]), align='center')
ax3.set_title('Color Histogram - Y ch', fontsize=15)
ax4.plot(spatial_car)
ax4.set_title('Spatially Binned Features', fontsize=15)
plt.axis('tight')
fig.savefig('output_images/color_spatial_feats_car.jpg', dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
            papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)

#######################
# III. Not-Car Features

im_notcar = mpimg.imread(notcars[np.random.randint(0, len(notcars))])
gray_notcar = cv2.cvtColor(im_notcar, cv2.COLOR_RGB2GRAY)
_, hog_im_notcar = get_hog_features(gray_notcar, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
ch1hist_notcar, _, _, bincen_notcar, _ = color_hist(cv2.cvtColor(im_notcar, cv2.COLOR_RGB2YCrCb),
                                                    nbins=hist_bins, bins_range=(0, 1))
spatial_notcar = bin_spatial(cv2.cvtColor(im_notcar, cv2.COLOR_RGB2YCrCb), size=spatial_size)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.imshow(im_notcar, cmap='gray')
ax1.set_title('Example of Not-Car Image', fontsize=15)
ax2.imshow(hog_im_notcar, cmap='gray')
ax2.set_title('Hog Visualization', fontsize=15)
fig.savefig('output_images/hog_features_nocar.jpg', dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
            papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)

fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
ax3.bar(bincen_notcar, ch1hist_notcar[0], width=0.75 * (bincen_notcar[1] - bincen_notcar[0]), align='center')
ax3.set_title('Color Histogram - Y ch', fontsize=15)
ax4.plot(spatial_notcar)
ax4.set_title('Spatially Binned Features', fontsize=15)
plt.axis('tight')
fig.savefig('output_images/color_spatial_feats_nocar.jpg', facecolor='w', edgecolor='w', orientation='portrait',
            papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)

########################################################
# IV. Bounding boxes annotated on the set of test-images

images = glob.glob('test_images/*.jpg')
out = []
for image in images:
    img = mpimg.imread(image)
    aux = np.copy(img)
    # Uncomment the following line if you extracted training data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    img = img.astype(np.float32)/255.

    windows = slide_variable_window(img, x_start, y_start, xy_window, x_overlap=x_overlap, y_overlap=y_overlap)

    hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space, spatial_size=spatial_size,
                                 hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(aux, hot_windows, color=(0, 0, 255), thick=6)
    out.append(window_img)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(14, 8))
ax1.imshow(out[0])
ax1.set_title(images[0].split('/')[-1], fontsize=15)
ax2.imshow(out[1])
ax2.set_title(images[1].split('/')[-1], fontsize=15)
ax3.imshow(out[2])
ax3.set_title(images[2].split('/')[-1], fontsize=15)
ax4.imshow(out[3])
ax4.set_title(images[3].split('/')[-1], fontsize=15)
ax5.imshow(out[4])
ax5.set_title(images[4].split('/')[-1], fontsize=15)
ax6.imshow(out[5])
ax6.set_title(images[5].split('/')[-1], fontsize=15)
plt.tight_layout()
fig.savefig('output_images/windows_on_testims.jpg', dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)
