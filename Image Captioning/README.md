<h1 align="center">Image Captioning</h1>

## Problem Statement: 

Annotate an image with a short description explaining the contents in that image.

## Dataset used:
The Microsoft Common Objects in COntext (MS COCO) dataset is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.  

You can read more about the dataset on the [website](http://cocodataset.org/#home)

## Project Planning:

The project is divided into the following tasks and, each task is carried out in it's respective Jupyter Notebook.
1. **Dataset Exploration:** In this notebook, the COCO dataset is explored, in preparation for the project.
2. **Preprocessing:** In this notebook, COCO dataset is loaded and pre-processed, making it ready to pass to the model for training.
3. **Training:** In this notebook, the CNN-RNN deep architecture model is trained.
4. **Inference:** In this notebook, the trained model is used to generate captions for images in the test dataset. Here, the performance of the model is observed on real world images.  

<img
  src="https://github.com/Praveen-Samudrala/Deep-Learning/blob/main/Image%20Captioning/Images/encoder-decoder.png"
  alt="Model Architecture"
  title="Model Architecture"
  style="display: inline-block; margin: 0 auto; width:800px; height:400px"/>


## Result
<img
  src="https://github.com/Praveen-Samudrala/Deep-Learning/blob/main/Image%20Captioning/Images/result1.png"
  alt="Result 1"
  title="Result 1"
  style="display: inline-block; margin: 0 auto; width:400px; height:300px"/>
  
  <img
  src="https://github.com/Praveen-Samudrala/Deep-Learning/blob/main/Image%20Captioning/Images/result2.png"
  alt="Result 2"
  title="Result 2"
  style="display: inline-block; margin: 0 auto; width:400px; height:300px"/>
  
  <img
  src="https://github.com/Praveen-Samudrala/Deep-Learning/blob/main/Image%20Captioning/Images/result3.png"
  alt="Result 3"
  title="Result 3"
  style="display: inline-block; margin: 0 auto; width:400px; height:300px"/>
  
  <img
  src="https://github.com/Praveen-Samudrala/Deep-Learning/blob/main/Image%20Captioning/Images/result4.png"
  alt="Result 4"
  title="Result 4"
  style="display: inline-block; margin: 0 auto; width:400px; height:300px"/>
  

## Licensing, Authors, Acknowledgements
Credits must be given to Udacity for supporting while project development.

## Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Pytorch with CUDA](https://pytorch.org/)
