# Thumbs Gesture Recognition Using Machine Learning

This beginner-friendly project classifies hand images into three classes
using an SVM (Support Vector Machine):

-   `thumbs_up`
-   `thumbs_down`
-   `neutral` --- a gesture that is neither thumbs-up nor thumbs-down

The project uses Python, Pillow, NumPy, Matplotlib, Scikit-learn, and
Joblib. Follow the steps in order.

> **Note:** Results depend on image quality and variety. The neutral
> class currently has fewer images than the other classes, so its
> predictions may be less reliable. Test with new images.

------------------------------------------------------------------------

## Step 1 --- Download the project files from GitHub

1.  Open this GitHub repository.
2.  Download `Thumbs_Gesture_Recognition_ML.ipynb`.
3.  Download `Thumbs_Gesture_Dataset.zip` if it is uploaded to the
    repository.
4.  Keep both files on your computer.

If the dataset ZIP is not present, ask the project owner to upload/share
it. You do not need to download images one by one.

## Step 2 --- Open the notebook in Google Colab

1.  Open https://colab.research.google.com/
2.  Select **File → Upload notebook**.
3.  Upload `Thumbs_Gesture_Recognition_ML.ipynb`.
4.  Run the notebook cells in order.

Files saved under `/content` are temporary and may disappear when the
Colab runtime resets.

## Step 3 --- Upload the dataset ZIP to Colab

Run this cell and select `Thumbs_Gesture_Dataset.zip`:

``` python
from google.colab import files
uploaded = files.upload()
```

## Step 4 --- Extract the dataset

``` python
import os
import zipfile

zip_path = "/content/Thumbs_Gesture_Dataset.zip"
extract_path = "/content/gesture_dataset"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted.")
print("Top-level contents:", os.listdir(extract_path))
```

The expected structure is:

``` text
/content/gesture_dataset/
├── thumbs_up/
├── thumbs_down/
└── neutral/
```

If the ZIP contains an extra parent folder, set `dataset_path` in Step 6
to the directory that directly contains these three class folders.

## Step 5 --- Check the dataset folders

``` python
import os

dataset_path = "/content/gesture_dataset"
classes = ["thumbs_up", "thumbs_down", "neutral"]

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_path):
        print(f"Folder missing: {class_path}")
        continue

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"{class_name}: {len(images)} images")
```

If a class folder is missing or has zero images, fix the folder path
before continuing.

## Step 6 --- Import libraries and load/preprocess images

``` python
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

dataset_path = "/content/gesture_dataset"
classes = ["thumbs_up", "thumbs_down", "neutral"]

image_features = []
image_labels = []

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_path):
        raise FileNotFoundError(f"Missing class folder: {class_path}")

    for filename in os.listdir(class_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(class_path, filename)
        try:
            with Image.open(image_path) as image:
                image = image.convert("L").resize((64, 64))
                features = np.asarray(image, dtype=np.float32).flatten() / 255.0

            image_features.append(features)
            image_labels.append(class_name)
        except Exception as error:
            print(f"Skipping {image_path}: {error}")

X = np.asarray(image_features, dtype=np.float32)
y = np.asarray(image_labels)

print("Feature matrix:", X.shape)
print("Labels:", y.shape)
print("Classes:", np.unique(y))
```

Each image becomes 4,096 numbers because `64 × 64 = 4096`. Grayscale
pixel values are divided by 255 so they are between 0 and 1.

## Step 7 --- Split into training and testing data

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))
```

The training data teaches the model. The test data estimates how it
performs on held-out images. If many images are near-identical frames
from one video/capture, an image-level random split can make accuracy
look overly optimistic. For a stronger evaluation, keep frames from the
same capture sequence in the same split.

## Step 8 --- Train the SVM model

``` python
svm_model = SVC(kernel="linear", C=1.0)
svm_model.fit(X_train, y_train)

print("SVM model training completed!")
```

An SVM learns decision boundaries that separate the labelled classes.

## Step 9 --- Evaluate the model

``` python
y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=classes, zero_division=0))

print("\nConfusion Matrix (rows = true class, columns = predicted class):")
print(confusion_matrix(y_test, y_pred, labels=classes))
```

Accuracy is the fraction of test images predicted correctly. Precision,
recall, F1-score, and the confusion matrix help identify which classes
are being confused. Results may vary with the dataset and split.

## Step 10 --- Test the model on a new image

Run this cell and upload a JPG or PNG image with a clearly visible hand:

``` python
from google.colab import files

uploaded_test = files.upload()
test_filename = next(iter(uploaded_test))
test_image_path = "/content/" + test_filename

with Image.open(test_image_path) as image:
    display_image = image.convert("RGB")
    processed_image = image.convert("L").resize((64, 64))
    test_features = np.asarray(processed_image, dtype=np.float32).flatten() / 255.0

prediction = svm_model.predict(test_features.reshape(1, -1))[0]

plt.figure(figsize=(4, 4))
plt.imshow(display_image)
plt.axis("off")
plt.title(f"Predicted gesture: {prediction}")
plt.show()

print("Predicted gesture:", prediction)
```

Try images with different backgrounds, lighting, distances, and hand
positions. If predictions are poor, collect more correctly labelled
images---especially for `neutral`---and retrain.

## Step 11 --- Save the trained model as a pickle file

``` python
model_path = "/content/thumbs_gesture_svm_model.pkl"
joblib.dump(svm_model, model_path)

print("Model saved to:", model_path)
print("File exists:", os.path.exists(model_path))
```

This saves the trained SVM model. It does not include the notebook or
the dataset.

## Step 12 --- Download the pickle file

``` python
from google.colab import files
files.download("/content/thumbs_gesture_svm_model.pkl")
```

Your browser should download `thumbs_gesture_svm_model.pkl`.

### Load the saved model later

After uploading the `.pkl` file into a new Colab session, run:

``` python
import joblib
svm_model = joblib.load("/content/thumbs_gesture_svm_model.pkl")
print("Saved SVM model loaded successfully!")
```

Preprocess new images in exactly the same way as in Step 10 before
predicting.

**Security warning:** Only load pickle/Joblib files from sources you
trust; loading an untrusted pickle file can execute harmful code.

## Step 13 --- Download the completed notebook

In Colab, wait for the notebook to save, then choose **File → Download →
Download .ipynb**. Keep this notebook alongside the downloaded model
file.

## Suggested GitHub repository structure

``` text
Thumbs-Gesture-Recognition/
├── README.md
├── Thumbs_Gesture_Recognition_ML.ipynb
├── requirements.txt
├── .gitignore
└── Thumbs_Gesture_Dataset.zip   # include if size and sharing permissions allow
```

Suggested `requirements.txt`:

``` text
numpy
matplotlib
Pillow
scikit-learn
joblib
```

Suggested `.gitignore`:

``` text
__pycache__/
.ipynb_checkpoints/
.env
venv/
.venv/
```

GitHub's browser upload has a per-file size limit. If the ZIP is too
large, use an approved file-sharing service or Git LFS instead of
repeatedly trying to upload it through the browser.

## Workflow summary

``` text
Download notebook and dataset ZIP
        ↓
Open notebook in Google Colab
        ↓
Upload and extract dataset
        ↓
Load and preprocess images
        ↓
Split training and test data
        ↓
Train SVM model
        ↓
Evaluate accuracy and inspect confusion matrix
        ↓
Test a new image
        ↓
Save model as .pkl
        ↓
Download the pickle file and completed notebook
```

## Learning reminder

Be ready to explain image preprocessing, why the data is split into
training and testing sets, what an SVM does, what the accuracy/confusion
matrix means, and how to load the saved model for later predictions.
