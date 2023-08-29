# CIFAR-10 Image Classification using CNN  
   
This project is focused on classifying the CIFAR-10 dataset using a Convolutional Neural Network (CNN) in TensorFlow. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.  
   
## Features  
   
- Data loading and preprocessing, including conversion of images to grayscale and normalization.  
- Visualization of images from the dataset.  
- One-hot encoding of target variables.  
- CNN model creation and training using TensorFlow.  
- Model evaluation using accuracy and loss plots.  
- Saving and loading the trained model.  
- Confusion matrix visualization for model evaluation.  
- Displaying actual vs. predicted labels for test images.  
   
## Installation  
   
To install the required packages, run the following command:  
   
```bash  
pip install numpy pandas matplotlib seaborn opencv-python tensorflow  
```  
   
## Usage  
   
1. Load the dataset using the provided code.  
2. Preprocess the dataset by converting the images to grayscale, normalizing, and one-hot encoding the target variables.  
3. Train the CNN model on the preprocessed dataset.  
4. Evaluate the model using accuracy and loss plots, confusion matrix, and actual vs. predicted labels for test images.  
5. Save the trained model for future use or load a previously saved model.  
   
## Example Use Case: Image Classification  
   
This project can be applied to various image classification tasks, such as object recognition, scene recognition, and image annotation. By training the CNN model on a specific dataset, it can learn to recognize patterns and features within the images, allowing it to classify new, unseen images accurately.  
   
For example, this project can be adapted to classify images of animals, vehicles, or other objects depending on the dataset used for training. The trained model can then be used in applications such as computer vision, robotics, or image search engines to automatically categorize and label images.