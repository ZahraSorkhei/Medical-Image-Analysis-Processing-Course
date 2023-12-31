# -*- coding: utf-8 -*-
"""MIAP_HW01_Q2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Sg68pDChe7IkiOasc5ijvuarBOzZuAop

### Q2
"""

import numpy as np
from PIL import Image, ImageOps
from numpy import asarray
from matplotlib import pyplot as plt
from scipy import ndimage

#Compute SNR as the formula in homwork
def snr(image,image_noise):
  x1 = np.sum(np.power(image,2))
  x2 = np.sum(np.power(image-image_noise,2))
  return 10*np.log10(x1/x2)

#Load original image and noisy image
image = asarray(ImageOps.grayscale(Image.open('city_orig.jpg')))
image_noise  = asarray(ImageOps.grayscale(Image.open('city_noise.jpg')))

#convert type of images to int
image = image.astype('int')
image_noise = image_noise.astype('int')

#split images to 4 part
M = image.shape[0]//2 -1
N = image.shape[1]//2 +1
image1 = image[0:M,0:N]
image2 = image[M:image.shape[0],0:N]
image3 = image[0:M,N:image.shape[1]]
image4 = image[M:image.shape[0],N:image.shape[1]]
image1_noise = image_noise[0:M,0:N]
image2_noise = image_noise[M:image.shape[0],0:N]
image3_noise = image_noise[0:M,N:image.shape[1]]
image4_noise  = image_noise[M:image.shape[0],N:image.shape[1]]

#Find SNR of each noisy part
snr_salt = snr(image1 , image1_noise)
snr_gaussian_salt =  snr(image2 , image2_noise)
snr_without_noise =  snr(image3 , image3_noise)
snr_gaussian =  snr(image4 , image4_noise)

print("SNR of with salt and pepper noise = ",snr_salt)
print("SNR of with gaussian noise = ", snr_gaussian)
print("SNR of with salt and paper and gaussian noise = ",snr_gaussian_salt)
print("SNR of without noise =  ",snr_without_noise)

def mean_filter(img):
      '''In this function we find the average of 9 pixels those are in 3*3 square '''
      m, n = img.shape
      mask = np.ones((3, 3), dtype = int)/9
      img_new = img - img
        
      for i in range(1, m-1):
          for j in range(1, n-1):
            temp = 0
            for k in range(3):
              for u in range(3):
                temp += img[i-1+k, j-1+u]*mask[0+k, 0+u]
            img_new[i, j]= temp
                
      img_new = img_new.astype(np.uint8)
      return img_new

def median_filter(img):
      '''In this function we find the median of 9 pixels those are in 3*3 square '''
      m, n = img.shape
        
      img_new = img - img
        
      for i in range(1, m-1):
          for j in range(1, n-1):
              temp = sorted([img[i-1, j-1],
                    img[i-1, j],
                    img[i-1, j + 1],
                    img[i, j-1],
                    img[i, j],
                    img[i, j + 1],
                    img[i + 1, j-1],
                    img[i + 1, j],
                    img[i + 1, j + 1]])
                
              img_new[i, j]= temp[4]
        
      img_new = img_new.astype(np.uint8)
      return img_new

def gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / g.sum()
def convolution(image, kernel):
    # Get the shape of the image and the kernel
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    # Define the output array
    output = np.zeros_like(image)

    # Pad the image with zeros to handle the edges
    pad_height = kernel_row // 2
    pad_width = kernel_col // 2
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    # Loop over every pixel of the image
    for row in range(image_row):
        for col in range(image_col):
            # Multiply the kernel with the image pixels and sum the result
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
    
    return output
def gaussian_filter(img):
  kernel = gaussian_kernel(5, 1)
  filtered_img = np.zeros_like(img)
  filtered_img = convolution(img, kernel)
  return filtered_img.astype(np.uint8)

# Test filters on image with salt and pepper noise
image1_gauss_filter = ndimage.gaussian_filter(image1_noise, sigma=1)
image1_mean_filter = mean_filter(image1_noise)
image1_median_filter =  ndimage.median_filter(image1_noise, size=3)

# Find SNR after filtering for image with salt and pepper noise
snr_salt_gauss_filter = snr(image1 , image1_gauss_filter)
snr_salt_mean_filter = snr(image1 , image1_mean_filter)
snr_salt_median_filter = snr(image1 , image1_median_filter)

# Test filters on image with salt and pepper and gaussian noises
image2_gauss_filter = ndimage.gaussian_filter(image2_noise, sigma=1)
image2_mean_filter = mean_filter(image2_noise)
image2_median_filter =  ndimage.median_filter(image2_noise, size=3)

# Find SNR after filtering for image with salt and pepper and gaussian noises
snr_gaussian_salt_gauss_filter = snr(image2 , image2_gauss_filter)
snr_gaussian_salt_mean_filter = snr(image2 , image2_mean_filter)
snr_gaussian_salt_median_filter = snr(image2 , image2_median_filter)

# Test filters on image with gaussian noise
image4_gauss_filter = ndimage.gaussian_filter(image4_noise, sigma=2)
image4_mean_filter = mean_filter(image4_noise)
image4_median_filter =  ndimage.median_filter(image4_noise, size=3)

# Test filters on image with gaussian noise
snr_gaussian_gauss_filter = snr(image4 , image4_gauss_filter)
snr_gaussian_mean_filter = snr(image4 , image4_mean_filter)
snr_gaussian_median_filter = snr(image4 , image4_median_filter)

#Show SNR in a tabel
from tabulate import tabulate
data = [['image salt and pepper noise', snr_salt, snr_salt_gauss_filter, snr_salt_mean_filter,snr_salt_median_filter,'Median'],
['image gaussian noise and salt and pepper noise', snr_gaussian_salt, snr_gaussian_salt_gauss_filter, snr_gaussian_salt_mean_filter,snr_gaussian_salt_median_filter,'Median'],
['image gaussian noise', snr_gaussian, snr_gaussian_gauss_filter, snr_gaussian_mean_filter,snr_gaussian_median_filter,'Gaussian']]
print (tabulate(data, headers=["name ", "SNR of noisy image", "SNR after gauss filter ", "SNR after mean filter ","SNR after median filter ","Visual"]))

fig = plt.figure(figsize=(20, 10))
rows = 4
columns = 5
fig.add_subplot(rows, columns, 6)
plt.imshow(image1,cmap='gray')
plt.axis('off')
plt.title("Original image")
fig.add_subplot(rows, columns, 7)
plt.imshow(image1_noise,cmap='gray')
plt.axis('off')
plt.title("image with salt and pepper")
fig.add_subplot(rows, columns, 8)
plt.imshow(image1_gauss_filter,cmap='gray')
plt.axis('off')
plt.title("gaussian filter")
fig.add_subplot(rows, columns, 9)
plt.imshow(image1_mean_filter,cmap='gray')
plt.axis('off')
plt.title("mean filter")
fig.add_subplot(rows, columns, 10)
plt.imshow(image1_median_filter,cmap='gray')
plt.axis('off')
plt.title("median filter")
#################################################
fig.add_subplot(rows, columns, 11)
plt.imshow(image2,cmap='gray')
plt.axis('off')
plt.title("Original image")
fig.add_subplot(rows, columns, 12)
plt.imshow(image2_noise,cmap='gray')
plt.axis('off')
plt.title("image with salt and pepper and gaussian noise")
fig.add_subplot(rows, columns, 13)
plt.imshow(image2_gauss_filter,cmap='gray')
plt.axis('off')
plt.title("gaussian filter")
fig.add_subplot(rows, columns, 14)
plt.imshow(image2_mean_filter,cmap='gray')
plt.axis('off')
plt.title("mean filter")
fig.add_subplot(rows, columns, 15)
plt.imshow(image2_median_filter,cmap='gray')
plt.axis('off')
plt.title("median filter")
#################################################
fig.add_subplot(rows, columns, 16)
plt.imshow(image4,cmap='gray')
plt.axis('off')
plt.title("Original image")
fig.add_subplot(rows, columns, 17)
plt.imshow(image4_noise,cmap='gray')
plt.axis('off')
plt.title("image with gaussian noise")
fig.add_subplot(rows, columns, 18)
plt.imshow(image4_gauss_filter,cmap='gray')
plt.axis('off')
plt.title("gaussian filter")
fig.add_subplot(rows, columns, 19)
plt.imshow(image4_mean_filter,cmap='gray')
plt.axis('off')
plt.title("mean filter")
fig.add_subplot(rows, columns, 20)
plt.imshow(image4_median_filter,cmap='gray')
plt.axis('off')
plt.title("median filter")

"""***Salt and pepper noise*** is a type of noise that affects only a few pixels in an image, but makes them very bright or very dark. This can degrade the quality and contrast of the image1.

To remove salt and pepper noise, different filters can be used, such as mean, median and Gaussian filters. However, they have different advantages and disadvantages.

Mean filter: This filter replaces each pixel with the average of its neighboring pixels. This can smooth out the noise, but also blur the edges and details of the image2.
Median filter: This filter replaces each pixel with the median of its neighboring pixels. This can preserve the edges and details better than the mean filter, but also introduce some artifacts and distortions2.


Gaussian filter: This filter replaces each pixel with a weighted average of its neighboring pixels, where the weights are determined by a Gaussian function. This can reduce the noise and preserve the edges and details better than the mean filter, but also blur the image more than the median filter3.


**According to a comparative analysis of different filters for salt and pepper noise removal in images, the median filter performs better than the mean and Gaussian filters.**

**based on SNR values, Median filter has performed better.**

***Gaussian noise*** is a type of noise that affects all the pixels in an image, but makes them slightly brighter or darker. This can reduce the sharpness and contrast of the image1.

To remove Gaussian noise, different filters can be used, such as mean, median and Gaussian filters. However, they have different advantages and disadvantages.


According to a comparative analysis of different filters for Gaussian noise removal, the best filter depends on the level of noise and the number of iterations of the filter. Sometimes the Gaussian filter is better and sometimes the median filter is better. The denoising autoencoder, which is a deep learning model, can also handle Gaussian noise well, but it takes more time than the other filters3. When considering only the time parameter, the median filter gives better results in less time than the Gaussian filter and the denoising autoencoder3.

**But based on SNR values, Gaussian filter has performed better.**

**For the mixed noise, it seems that the median filter causes less blurring but Gaussian filter result has less noise, at the same time, according to the SNR values, the Gaussian filter has performed better.**
"""