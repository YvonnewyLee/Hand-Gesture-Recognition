# Hand Gesture Recognition README

## Project Overview
This project focuses on recognizing hand gestures representing the American Sign Language (ASL) alphabet for the letters A-I. The task is to train a Convolutional Neural Network (CNN) to classify images of hand gestures. The project is divided into two parts:

- **Part A**: Data Collection and Preprocessing
- **Part B**: Model Building, Training, and Hyperparameter Tuning

## Project Structure

### Part A: Data Collection & Preprocessing
In Part A, I collected and cleaned the dataset by following these steps:

1. **Data Collection**: I captured 3 images for each hand gesture representing the letters A-I in American Sign Language.
2. **Data Cleaning**: The images were standardized to a size of 224x224 pixels (RGB). The hand gesture is centered, and the background is kept consistent (white wall).

### Part B: Model Building & Training
Part B includes building and training a Convolutional Neural Network (CNN) and utilizing Transfer Learning.

#### 1. **Data Loading & Splitting**
The dataset is split into training, validation, and test sets based on the student who captured the images. This ensures that images from the same student don't overlap in different sets, preventing data leakage.

#### 2. **Building the CNN Model**
A custom CNN architecture is designed with two convolutional layers, two pooling layers, and two fully connected layers to predict the ASL letters (A-I).

#### 3. **Training & Evaluation**
The model is trained using **Cross-Entropy Loss** and **Stochastic Gradient Descent (SGD)** optimizer. The performance of the model is evaluated on the training, validation, and test sets.

#### 4. **Hyperparameter Tuning**
I explored hyperparameter tuning for **batch size**, **number of epochs**, and **kernel size** to improve the model's performance. Training accuracy, validation accuracy, and loss curves are plotted for various hyperparameter settings.

#### 5. **Transfer Learning**
To improve the model’s performance, I applied Transfer Learning using a pre-trained **AlexNet** model. The convolutional layers of AlexNet are used to extract features, which are then passed through a custom classifier to predict hand gestures.

### Results
- **Custom CNN**: The model achieved a test accuracy of 57.86% on the unseen test data and 44.44% on my personal hand gestures.
- **Transfer Learning with AlexNet**: The transfer learning model achieved a test accuracy of 76.06% on the test data and 100% accuracy on my sample images.

### Conclusion
- **Custom CNN**: Struggled to generalize to new, unseen data due to overfitting and a small dataset.
- **Transfer Learning with AlexNet**: Performed significantly better, achieving high accuracy even on my personal sample images, demonstrating the benefits of using a pre-trained model for feature extraction.

### Next Steps for Improvement
- **Data Augmentation**: To further improve model generalization, I will use transformations like rotation, scaling, and translation.
- **Larger Dataset**: Collecting more images with messy backgrounds and varying conditions will help the model handle real-world scenarios better.
- **Regularization**: Techniques like dropout and weight decay will be applied to prevent overfitting.


## Usage
- **Part A**: Data preprocessing and augmentation are handled in the respective Python scripts.
- **Part B**: The CNN training and evaluation scripts allow you to train the model using the prepared dataset and fine-tune hyperparameters.
