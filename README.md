# machine_learning_assignments
Repository of machine learning assignments

Major guidelines:
- Upto Assignment 5 (No AI is used, most barebone coding with building many functions from scratch into modules, which are reused.)
- Assignment 6 onwards are not that pure(used ai assistance for getting torch modules).

## Assignment 1
Image Processing
- Adjusting Brightness
- Image Reflection operation
- From a real image of circle, predict the actual radius of it after processing
- Calculation to get Focal length of the camera

## Assignment 2
- DPI calculation
- Algorithm to calculate number of lines from document
- Thresholding algorithms on image
    - binary thresholding
    - Otsu thresholding
 
- Histogram equalization, Differential Probability histogram


## Assignment 3
- Applying filters on image
    - Mean (3x3)
    - Median (3x3)
    - Gaussian (5x5) with sigma = 2
    - Sharpening filter
      (No function calls)

## Assignment 4
- Applying filters on image
  - Robert
  - Prewitt
  - Sobel
  - Laplacian with Gaussion (LoG)
  - Canny Edge Detection
    1) Gaussian Filter
    2) Gradient+angles calculation
    3) Non-Maxima Suppression
    4) Double thresholding ( getting weak and strong edges)
    5) Edge tracking via Hysterisis
   
## Assignment 5
- From a picture of triangle inside a square,
  1) Identify the straight lines via Hough's transformations
  2) Find absolute area of triangle in cm^2
  3) Rotate and skew the image
 
# Neural Networks Training 

## Assignment 6: FCN to identify numbers from 0 to 9
1) Fully connected network to identify numbers from 0 to 9
2) Evaluation metrics and output accuracy
3) Store weights and biases into separate files
4) From image of an arbitrary fence, create one random dot per image. Organize the images into those with dot inside the fence or outside the fence.

## Assignment 7: FCN 
1) FNC to identify the ball is inside or outside the fence.(from prev assignment images for training)
2) Check performance
3) Create a Binary Classifier FNC to classify randomly generated matrices as valid or invalid. (refer materials to know what an invalid matrix is)
4) Get evaluation metrics - F1 score, precision, recall, accuracy
5) Doing this for: 
   a) 10000(5000 valid + 5000 invalid) 2x2 matrices
   b) 1000 random 2x2 matrices

## Assignment 8: CNN
1) For fence-dot/ball detection, create a CNN model instead of FCN model.
2) Comparing the performance of CNN and FCN
3) Use CNN on MNIST dataset and check the accuracy (only less epochs needed for CNN)

## Assignment 9: YOLOv8
1) Using COCO pre-trained YOLOv8 model, check the performance
2) Take a video footage of 30 sec, extract each frames, annotate them using labelme(will give as json files with object bounding boxes/any shapes). Save the annotation. (This can later be used in training models to identify this object)

## Assignment 10: YOLO & GNN (Regression/classification problem)
1) Train YOLOv8 on VOC2012 dataset. Evaluate it's performance
2) Using pytorch geometric, load MUTAG dataset.
    - Visualize molecular graphs of mutagenic or non-mutagenic examples.
  
3) Design and train Graph Neural Network(GNN) of our choice to classify molecules into mutagenic or non-mutagenic
4) For each molecules in the dataset, predict number of 6 membered rings per molecule.
5) Report appropriate evaluation metrics for GNN performance.




