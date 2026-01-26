---
alts:
tags:
  - ML
date: 2026-01-24 17:51
---
## Theoretical Workout
In Problem 04 of Assignment_01, ﬁnd out the number of pixels per square inch (DPI) for both the images taken at 50 cm and 80 cm respectively.

![[ml.assignment2 2026-01-22 18.54.05.excalidraw|100%]]

> [!abstract] 
> 1. finding dpi
> 2. Reading number of lines
> 3. Diff thresholding algorithm implement
> 4. Filter apply, thresholding, draw random circles
> 5. Equalization implementation
## Concepts
![[ml.assignment2 2026-01-25 11.32.56.excalidraw|100%]] 

## otsu's thresholding
So basically otsu's thresholding is a clusteing based method. Assumes the image contains foreground and background pixels of the image. 
Calculating optimal threshold that separate the two -> 
by increasing intra-class variance and reducing inter-class variance.
![[ml.assignment2 2026-01-26 11.30.18.excalidraw|100%]]

## Question 4

Draw a fence of an arbitrary shape by hand (!) on a clean paper with white background and printed lines on it as explained in the class. 
Take a photo of this image. 
Write a program: 
a) to read the photographed image and translate it to a gray scale image of size say, 300 x 300 of this image. 
b)Use global thresholding algorithm to clean image to remove all unwanted lines on it if any. 
c) Save original and binary image by name fence_original.jpg and fence_threshold.jpg. 
d) Write a program to draw a ﬁlled circle of radius 5 randomly in this image (fence_threshold.jpg). Generate 50 such images. Store these images into two folders as explained in the class.

![[ml.assignment2 2026-01-26 18.50.24.excalidraw|100%]]
___
