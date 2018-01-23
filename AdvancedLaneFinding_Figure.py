# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:37:32 2018

@author: Jie Wang

#%%
Project Steps
https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/44732d48-dcfe-4b4e-9614-12422ec29306

First steps:

Camera calibration
Distortion correction
Color/gradient threshold
Perspective transform

After doing these steps, two additional steps for the project:

Detect lane lines
Determine the lane curvature
"""
#%%
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%%
# ====================
# CAMERA CALIBRATION
# ====================

# prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0), ..., (8, 5, 0)
objp = np.zeros((6*9,3), np.float32)

# https://stackoverflow.com/questions/42308270/python-numpy-mgrid-and-reshape
# http://louistiao.me/posts/numpy-mgrid-vs-meshgrid/
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal\calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    # Convert images to grayscale in order to use cv2.findChessboardCorners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    # If corners are found, append those points to the image points arrary
    # The prepared points, objp, are added to the object points arrary. 
    # These object points will be the same for all the calibration images since they represent a real chessboard
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    #print(ret)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #cv2.imshow('img', img)
        #plt.imshow(img)
#%%
# we have objpoints and imgpoints needed for camera calibration. 
# now to calculate the camera matrix and distortion coefficients, 
# and then to test undistortion calibration on images
# Read in an image

#img_size = (img.shape[1], img.shape[0])
# TODO: Write a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted destination image (dest)
def cal_undistort(img, objpoints, imgpoints):
    img_size = (img.shape[1], img.shape[0])
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    return dst
    #return dst, dist, mtx
img = cv2.imread('camera_cal/calibration2.jpg')    
dest = cal_undistort(img, objpoints, imgpoints)
#dst, dist, mtx = cal_undistort(img, objpoints, imgpoints)

# cv2.imwrite('output_images/calibration2_undist.jpg',dest)

"""
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
#pickle.dump( dist_pickle, open( "calibration_wide/wide_dist_pickle.p", "wb" ) )
"""
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=25)
ax2.imshow(dest)
ax2.set_title('Undistorted Image', fontsize=25)     

#%%
# ========================
# DISTORATION CORRECTION
# ========================
test_imgs = mpimg.imread('test_images/straight_lines2.jpg')
#plt.imshow(test_imgs)
undist_test_imgs = cal_undistort(test_imgs, objpoints, imgpoints)  
#undist_test_imgs = np.asarray(list(map(lambda img: cal_undistort(img, objpoints, imgpoints), test_imgs)))

# cv2.imwrite('output_images/test2_undist.jpg',undist_test_imgs)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(test_imgs)
ax1.set_title('Original Image', fontsize=25)
ax2.imshow(undist_test_imgs)
ax2.set_title('Undistorted Image', fontsize=25) 
#%%
# ========================
# Color/gradient threshold
# ========================

# hls color threshold
def hls_white_yellow_binary(img):
    # Transfer the RGB to HLS (mpimg.imread was used)
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # yellow hls binary threshold (90, 255)
    """
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    """
    hls_yellow_binary = np.zeros_like(hls_img[:,:,2])
    hls_yellow_binary[(hls_img[:,:,2] >= 90) & (hls_img[:,:,2] <= 255)] = 1
    
    # white hls binary threshold (200, 255)
    # white is only determined by lightness, hue and saturation are not
    hls_white_binary = np.zeros_like(hls_img[:,:,1])
    hls_white_binary[(hls_img[:,:,1] >= 200) & (hls_img[:,:,1] <= 255)] = 1
    
    # Now combine both
    hls_binary = np.zeros_like(hls_img[:,:,1])
    hls_binary[(hls_yellow_binary == 1) | (hls_white_binary == 1)] = 1

    return hls_binary

hls_bin = hls_white_yellow_binary(undist_test_imgs)

# https://stackoverflow.com/questions/37026582/saving-an-image-with-imwrite-in-opencv-writes-all-black-but-imshow-shows-correct/37027314
# plt.imsave('output_images/test2_hls.jpg',hls_bin)


# Visualization 
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(undist_test_imgs)
ax1.set_title('Undistorted Image', fontsize=25)
ax2.imshow(hls_bin, cmap='gray')
ax2.set_title('HLS Thresholding Mask', fontsize=25) 

#%%
# gradient threshold

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8-bit (0 - 255) and convert to type np.uint8
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return the binary image
    return binary_output

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# # apply gradient threshold on grayscale image 
img = undist_test_imgs
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Apply each of the thresholding functions
gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(50, 225))
grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(50, 225))
mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=(20, 150))
dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))

grad_combined = np.zeros_like(dir_binary)
grad_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

#plt.imsave('output_images/test2_grad.jpg',grad_combined)

# Visualization 
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Undistorted Image', fontsize=25)
ax2.imshow(grad_combined, cmap='gray')
ax2.set_title('Gradient Thresholding Mask', fontsize=25) 

#%%
# combine color and gradient threshold 
# Combine the two binary thresholds
combined_binary = np.zeros_like(grad_combined)
combined_binary[(grad_combined == 1) | (hls_bin == 1)] = 1

#plt.imsave('output_images/test2_thresh.jpg',combined_binary)

# Visulization
# Stack each channel to view their individual contributions in green and blue respectively
# This returns a stack of the two binary images, whose components you can see as different colors
hls_color_binary = np.dstack(( np.zeros_like(grad_combined), grad_combined, hls_bin)) * 255

# This method comes from 
# https://discussions.udacity.com/t/unable-to-draw-rectangles-on-curve-on-images/244276/61?u=jiewang
hls_color_binary = np.uint8(hls_color_binary)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('The gradient threshold (green) & the color threshold (blue)')
ax1.imshow(hls_color_binary)

ax2.set_title('the combined color and gradient thresholds')
ax2.imshow(combined_binary, cmap='gray')
#%%
# =====================
# Perspective transform
# =====================

img = undist_test_imgs

# define calibration box in source (orignial) and destionation (desired or warped) coordinate
src_pts = np.array([[690, 450], [1110,img.shape[0]-2],[210,img.shape[0]-2], [595, 450]], np.int32)
dst_pts = np.array([[1000,0], [1000,img.shape[0]-2],[200,img.shape[0]-2], [200, 0]], np.int32)

"""
# plot the sources points 
plt.plot(795, 515, '.') # top right 
plt.plot(1110, img.shape[0]-2, '.') # bottom right 
plt.plot(195,img.shape[0]-2, '.') # bottom left 
plt.plot(495, 515, '.') # top left 
"""

# define perspective tranform function
def warp(img, src_pts, dst_pts, flag = False):
   
    # four source coordinates
    src = src_pts.astype(np.float32)
    #four desired coordinates
    dst = dst_pts.astype(np.float32) 
    
    # compute the perspective transform matrix M for 2d images
    if flag == False:
        M = cv2.getPerspectiveTransform(src, dst)
        # create warped image using linear interpolation
        img_size = (img.shape[1], img.shape[0])
        
    # compute the perspective transform matrix M for 3d images
    elif flag == True:
        M = cv2.getPerspectiveTransform(dst, src)
        # create warped image using linear interpolation
        img_size = (img.shape[:2][1], img.shape[:2][0])
    
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped

# Put the threshold binary and perspective tranform together 
warped_im = warp(combined_binary , src_pts, dst_pts, flag = False)

# Visualization 
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
cv2.polylines(img,[src_pts],True,(255,0,0), 10)
ax1.imshow(img)
ax1.set_title('Warped Image', fontsize=25)
ax2.imshow(warped_im, cmap='gray')
ax2.set_title("Perspective Transform Image", fontsize=25) 

#%%
"""
After applying calibration, thresholding, and a perspective transform to a road image, you should 
have a binary image where the lane lines stand out clearly. However, you still need to decide 
explicitly which pixels are part of the lines and which belong to the left line and which belong to 
the right line.
"""
# =====================
# Detect Lane Lines
# =====================

# Take a histogram of the bottom half of the image
histogram = np.sum(warped_im[warped_im.shape[0]//2:,:], axis=0)

fig, ax = plt.subplots(1, 2, figsize=(15,4))
ax[0].imshow(warped_im, cmap='gray')
ax[0].set_title("Perspective Transform Image")
ax[1].plot(histogram)
ax[1].set_title("Histogram Of Pixel Intensity")
plt.show()

# This method comes from 
# https://discussions.udacity.com/t/unable-to-draw-rectangles-on-curve-on-images/244276/61?u=jiewang
warped_im = np.uint8(warped_im) 

# Create an output image to draw on and visualize the result
out_img = np.dstack((warped_im, warped_im, warped_im))*255

# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint]) # np.argmax returns the indices of the maximum values along an axis
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(warped_im.shape[0]/nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = warped_im.nonzero() # np.nonzero return the indices of the elements that are non-zero.
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = warped_im.shape[0] - (window+1)*window_height
    win_y_high = warped_im.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
    (0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
    (0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds) # np.concatenate join a sequence of arrays along an existing axis
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

#%%
# Visualization
# Generate x and y values for plotting
ploty = np.linspace(0, warped_im.shape[0]-1, warped_im.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)

#%%
# Visulizarion 2
# Create an image to draw on and an image to show the selection window
out_img = np.dstack((warped_im, warped_im, warped_im))*255
window_img = np.zeros_like(out_img)
# Color in left and right line pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)

#%%
# =============================
# Calculation of Lane Curvature
# =============================

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

# Define y-value where we want radius of curvature
# I'll choose the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(ploty)

# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# average radius of curvature is in meters
rad_curverad = (left_curverad + right_curverad)/2

# calculate the car offset
line_ave = np.mean(((right_fitx) + (left_fitx))/2)
line_offset = line_ave - (warped_im.shape[-1]//2)
car_offset = line_offset * xm_per_pix

# print out 
car_offset = 'Car Offset: ' + '{0:.2f}'.format(car_offset) + 'm'
rad_curverad = 'Curvature Radius:' + '{0:.2f}'.format(rad_curverad) + 'm'

# %%

# Create an image to draw the lines on
warp_zero = np.zeros_like(warped_im).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp =  warp(color_warp, src_pts, dst_pts, flag = True)
# Combine the result with the original image
result = cv2.addWeighted(undist_test_imgs, 1, newwarp, 0.3, 0)

cv2.putText(result, car_offset , (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), thickness=2)
cv2.putText(result, rad_curverad , (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), thickness=2)

plt.imshow(result)