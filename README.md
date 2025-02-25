# ASL-Hand-Gesture-Recognition

## Project Overview

This project is designed to recognize American Sign Language (ASL) hand gestures for letters A to I using Deep Learning.  
The model employs Transfer Learning with AlexNet for feature extraction and a CNN Classifier for final prediction.   
The project is implemented using PyTorch and deployed on Hugging Face Spaces for real-time ASL hand gesture classification.  

Try it here: Hugger Face Link:
https://huggingface.co/spaces/serenazheng/ASL_hand_gesture_recognition_A-I  

## Project Structure
ASL-Hand-Gesture-Recognition/  
│── data/  
│   ├── A2_Hand_Gesture_Dataset_revised.zip  # Dataset with ASL hand gesture images  
│   ├── test.zip  # Test dataset 27 images total, 3 images per class  
│   ├── textures.zip  # Background textures for data augmentation  
│── ASL_Hand_Gesture.ipynb  # Jupyter Notebook for model training & evaluation  
│── README.md  # Project documentation  
│── .gitignore  # Files to ignore in Git  

## Dataset Details

The dataset contains labeled images of ASL hand gestures for letters A to I. The dataset structure includes:

Dataset: over 2000 images unaugmented ASL hand gesture images for letters A to I. Split for training and validation.
Test Set: Contains 27 images (3 per class) for final model evaluation.
Textures Dataset: Background textures used for data augmentation.

## Data Preprocessing & Augmentation  
To improve model generalization, the following augmentations were applied:  
Rotation, Scaling, Flipping: Introduce variance in hand positioning.  
Background Texture Addition: Simulates diverse environments.  
Normalization: Standardized pixel values for improved training stability.  

## Model Architecture: Transfer Learning with CNN Classifier
🔹 Step 1: Feature Extraction with AlexNet  
AlexNet, a well-established Convolutional Neural Network (CNN) pre-trained on ImageNet, is used as a feature extractor in this project. Instead of training a model from scratch, we take advantage of AlexNet's early convolutional layers, which are excellent at capturing essential spatial patterns such as edges, shapes, and textures.  
🔹 Step 2: Custom CNN Classifier  
To adapt the extracted features to ASL hand gesture classification, we design a custom Convolutional Neural Network (CNN). This classifier refines the feature maps from AlexNet and maps them to 9 different ASL gestures.  
🔹 Step 3: Training the Model  
Once the model architecture is defined, we train it using the extracted features from AlexNet. The classifier is trained using a CrossEntropyLoss function, and the weights are updated using the Adam optimizer.  
🔹 Step 4: Hyperparameter Tuning
The best-performing model was obtained through hyperparameter tuning. The final configuration is as follows:  
Conv1 Channels: 256  
Conv2 Channels: 512  
FC1 Units: 256  
FC2 Units: 128  
Kernel Size: 7  
Learning Rate: 0.0001  
🔹 Step 5: Model Evaluation  
After training and hyperparameter Tuning, the model is evaluated on a test set using accuracy. A confusion matrix is used to visualize the model’s classification performance per class.  

## Model Performance Summary
Dataset	Accuracy  
Training	91.45%  
Validation	78.90%  
Test	92.59%  

## Future Improvements  
🚀 Enhancing Model Performance  
Replace AlexNet with ResNet / MobileNet for better feature extraction.  
Fine-tune hyperparameters using Bayesian Optimization.  
📸 Real-Time Gesture Recognition  
Integrate OpenCV for real-time ASL recognition.  
🌎 Deployment  
Deploy the model using Flask or FastAPI to serve predictions via a web app.


