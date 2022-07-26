<h1 align="center">Facial Keypoint Detection</h1>

## Project Aim:
Identify facial landmarks of human face, outlining the person's eyebrows, eyes, nose, lips and jawline.

## Project Outline:
In this project I have built a Convolutional Neural Network(CNN) based Deep Learning system to identify Facial Keypoints. The system takes in any image with faces, and highlights facial landmarks namely eyebrows, eyes, nose, lips and jawline for each detected face with 68 marker points.

The project workflow is split into 2 Jupyter Notebooks and a couple of python files.  
* Define Netowrk Architecture : Loading Data and Training the Convolutional Neural Network (CNN) model to locate the Facial Keypoints.
    1. Fetching Data
    2. Data Transformation
    3. Batching and Loading Data
    4. Training
    5. Testing
    6. Feature Visualization

* Facial Keypoint Detection Complete Pipeline : Building a pipeline from taking input image to highlight Facial Keypoints.
    1. Loading Model
    2. Importing image
    3. Extracting Faces
    4. Identify facial keypoints
  
* models.py     : Define the Convolutional Neural Network architecture.
* data_loader.py: Containes the images transformations necessary for the project.


## Facial Keypoints:

The facial landmarks identified in this project are eyebrows, eyes, nose, lips and jawline. They are highlighted with dots that are mapped on each human face in the image. These keypoints are used for a variety of tasks, such as face filters, emotion recognition, pose recognition, etc. 
In the below image, these keypoints are numbered, and you can see that specific ranges of points match different portions of the face.

<img src="https://github.com/Praveen-Samudrala/Deep-Learning/blob/main/Facial%20Keypoint%20Detection/Images/landmarks_numbered.jpg" width=400>

## Dataset:

[YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/) is used in this project, which contains images extracted from the videos of people in YouTube videos. The data for this project has been loaded from a S3 bucket.

## Result:
Input Image
<p><img
  src="https://github.com/Praveen-Samudrala/Deep-Learning/blob/main/Facial%20Keypoint%20Detection/Images/obamas.jpg"
  alt="Input Image"
  title="Input Image"
  style="display: inline-block; margin: 0 auto; width:500px; height:300px"/></p>
<b>Faces detected with keypoints marked on each of the detected face</b> <br>
<p><img
  src="https://github.com/Praveen-Samudrala/Deep-Learning/blob/main/Facial%20Keypoint%20Detection/Images/obamas_detected.png"
  alt="Faces Detected"
  title="Faces Detected"
  style="display: inline-block; margin: 0 auto; width:500px; height:300px"/></p>
  
<p>Identified Facial Landmarks</p>
<p><img
  src="https://github.com/Praveen-Samudrala/Deep-Learning/blob/main/Facial%20Keypoint%20Detection/Images/barack.png"
  alt="Barrack Obama"
  title="Barrack Obama"
  style="display: inline-block; margin: 0 auto; width:300px; height:300px"/></p>

<p><img
  src="https://github.com/Praveen-Samudrala/Deep-Learning/blob/main/Facial%20Keypoint%20Detection/Images/michelle.png"
  alt="Michelle Obama"
  title="Michelle Obama"
  style="display: inline-block; margin: 0 auto; width:300px; height:300px"/></p>
 
 ## Licensing, Authors, Acknowledgements
Credits must be given to Udacity for supporting while project development.

## Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Pytorch with CUDA](https://pytorch.org/)

