# Hand Gesture Recognition Using Machine Learning

## 1. Project Overview

This project uses machine learning to classify hand gestures into
three categories:

- Thumbs Up
- Thumbs Down
- Neutral

The model uses Histogram of Oriented Gradients (HOG) for feature
extraction and Support Vector Machine (SVM) for classification.

## 2. Objectives

- Build a basic hand gesture classification system.
- Collect images using a webcam.
- Prepare and preprocess image data.
- Train and evaluate an SVM classifier.
- Test predictions using a new webcam image.
- Understand the machine learning workflow.

## 3. Technologies Used

- Python
- Google Colab
- NumPy
- Matplotlib
- Pillow
- Scikit-learn
- Scikit-image
- Joblib
- Git and GitHub

## 4. Project Structure

    Thumbs-Gesture-Recognition/
    ├── Thumbs_Gesture_Recognition_ML.ipynb
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    └── models/
        └── gesture_recognition_model.pkl

## 5. Machine Learning Workflow

1. Collect hand gesture images.
2. Organize images into three classes.
3. Resize images to 64 x 64 pixels.
4. Convert images to grayscale.
5. Extract image features.
6. Train an SVM classifier.
7. Evaluate predictions using test data.
8. Test a new webcam image.

## 6. Algorithms

### Support Vector Machine

SVM is used to classify images into the three gesture categories.

### Histogram of Oriented Gradients

HOG extracts local edge and shape information from images.
These features are then supplied to the SVM classifier.

## 7. Experimental Results

The HOG + SVM model achieved 83.33% accuracy on the current
test split.

This is an initial experimental result. Further evaluation
using independent image sequences and varied backgrounds
is needed to estimate real-world performance reliably.

## 8. Running the Project

1. Open the notebook in Google Colab.
2. Install the required libraries if necessary.
3. Prepare the dataset.
4. Run the notebook cells in order.
5. Train the model.
6. Evaluate the results.
7. Test a new webcam image.

## 9. Team Collaboration

Git is used for version control.

GitHub is used to store the project, document changes,
and collaborate with teammates through branches and merges.

## 10. Future Improvements

- Collect more varied images.
- Improve hand-region detection.
- Evaluate on independent image sequences.
- Test with different people and backgrounds.
- Explore real-time webcam predictions.

## 11. Dataset Acknowledgment

The project uses webcam images and selected images from a
public hand-gesture dataset.

The original dataset should be credited to its creators,
and its license and redistribution conditions should be
followed.