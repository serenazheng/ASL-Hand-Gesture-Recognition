# ASL-Hand-Gesture-Recognition

This project is designed to recognize American Sign Language (ASL) hand gestures using Deep Learning. It employs:

Transfer Learning (AlexNet) to extract high-level features.  
A Custom CNN Model for final classification.  

The model is implemented using PyTorch and trained on a dataset of labeled ASL hand gesture images from A to I.  

Hugger Face Link:
https://huggingface.co/spaces/serenazheng/ASL_hand_gesture_recognition_A-I

📂 Project Structure
ASL-Hand-Gesture-Recognition/
│── data/
│   ├── A2_Hand_Gesture_Dataset_revised.zip  # Dataset with ASL hand gesture images
│   ├── test.zip  # Test dataset 27 images total, 3 images per class
│   ├── textures.zip  # Background textures for data augmentation
│── ASL_Hand_Gesture.ipynb  # Jupyter Notebook for model training & evaluation
│── README.md  # Project documentation
│── .gitignore  # Files to ignore in Git

Model Architecture: Transfer Learning with Custom CNN  
🔹 Step 1: Feature Extraction with AlexNet  
AlexNet, a well-established Convolutional Neural Network (CNN) pre-trained on ImageNet, is used as a feature extractor in this project. Instead of training a model from scratch, we take advantage of AlexNet's early convolutional layers, which are excellent at capturing essential spatial patterns such as edges, shapes, and textures.
🔹 Step 2: Custom CNN Classifier  
To adapt the extracted features to ASL hand gesture classification, we design a custom Convolutional Neural Network (CNN). This classifier refines the feature maps from AlexNet and maps them to 9 different ASL gestures.
🔹 Step 3: Training the Model  
Once the model architecture is defined, we train it using the extracted features from AlexNet. The classifier is trained using a CrossEntropyLoss function, and the weights are updated using the Adam optimizer.
🔹 Step 4: Model Evaluation  
After training, the model is evaluated on a separate test set using accuracy. A confusion matrix is used to visualize the model’s classification performance.

🏆 Model Performance Summary
Dataset	Accuracy  
Training	91.45%  
Validation	78.90%  
Test	92.59%  
The training accuracy is 91.45%, indicating effective learning.  
The test accuracy is 92.59%, demonstrating strong generalization to unseen data.  
The validation accuracy (78.90%) shows that minor improvements could be made to prevent overfitting.  

💡 Future Improvements  
🚀 Enhancing Model Performance  
Replace AlexNet with ResNet / MobileNet for better feature extraction.  
Fine-tune hyperparameters using Bayesian Optimization.  
📸 Real-Time Gesture Recognition  
Integrate OpenCV for real-time ASL recognition.  
🌎 Deployment  
Deploy the model using Flask or FastAPI to serve predictions via a web app.
