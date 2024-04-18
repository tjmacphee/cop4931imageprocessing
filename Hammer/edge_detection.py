import cv2
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Function to preprocess an image, detect edges, find contours, and extract Hu Moments features
def preprocess_and_extract_features(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Convert the image to grayscale
    # Convert the image to grayscale
    p = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    p = cv2.GaussianBlur(p, (13, 13), 0)
    p = cv2.medianBlur(p, 11)
    _, p = cv2.threshold(p, 250, 255, cv2.THRESH_BINARY_INV)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    k3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    p = cv2.erode(p, k1)
    p = cv2.copyMakeBorder(
        p,
        top=25,
        bottom=25,
        left=25,
        right=25,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, k3)
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, k2)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, k3)
    q = cv2.erode(p, k4)

    # Use Canny edge detection to detect edges
    #

    edges = cv2.Canny(q, 200, 255)

    #cv2.imshow(image_path + '_edges', edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract Hu Moments features from contours
    features = []
    for contour in contours:
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()  # Flatten to ensure a 1D array
        hu_moments = [-1 * np.log(abs(hu)) if hu != 0 else 0 for hu in hu_moments]
        features.append(hu_moments)


    # Pad features with zeros if fewer than 10 contours are detected
    num_contours = len(features)
    if num_contours < 10:
        num_zeros_to_pad = 10 - num_contours
        features.extend([[0] * 7] * num_zeros_to_pad)

    return features[:10]  # Limit to 10 contours and return


# Function to process all images in a folder and extract features
def process_images_in_folder(folder_path):
    # Initialize lists to store features and labels
    features = []
    labels = []

    # Process each image in the folder and extract features
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Check if the file is an image
            image_path = os.path.join(folder_path, filename)
            image_features = preprocess_and_extract_features(image_path)
            features.extend(image_features)
            labels.extend([1] * len(image_features))  # Assuming all images in the folder contain hammers

    return np.array(features), np.array(labels)


# Path to the folder containing hammer images for training
training_folder_path = 'images\\train'

# Process images in the training folder and extract features
X_train, y_train = process_images_in_folder(training_folder_path)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Path to the folder containing test images
test_folder_path = 'images\\test'


# Function to detect hammers in test images and draw bounding boxes
def detect_hammers_and_draw_boxes(folder_path, classifier):
    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Check if the file is an image
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            # Convert the image to grayscale
            # Convert the image to grayscale
            p = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            p = cv2.GaussianBlur(p, (13, 13), 0)
            p = cv2.medianBlur(p, 11)
            _, p = cv2.threshold(p, 250, 255, cv2.THRESH_BINARY_INV)
            k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
            k3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            p = cv2.erode(p, k1)
            p = cv2.copyMakeBorder(
                p,
                top=25,
                bottom=25,
                left=25,
                right=25,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )
            p = cv2.morphologyEx(p, cv2.MORPH_OPEN, k3)
            p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, k2)
            p = cv2.morphologyEx(p, cv2.MORPH_OPEN, k3)
            q = cv2.erode(p, k4)


            # Use Canny edge detection to detect edges
            edges = cv2.Canny(q, 200, 255)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_hammer = False

            # Iterate over the contours and classify each one
            for contour in contours:
                moments = cv2.moments(contour)
                hu_moments = cv2.HuMoments(moments).flatten()  # Flatten to ensure a 1D array
                hu_moments = [-1 * np.log(abs(hu)) if hu != 0 else 0 for hu in hu_moments]
                prediction = classifier.predict(np.array([hu_moments]))[0]  # Convert to 2D array and get first element
                if prediction == 1:  # If the contour is classified as a hammer
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green bounding box
                    detected_hammer = True

            if not detected_hammer:
                cv2.putText(image, 'Did not detect Hammers', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the image with bounding boxes
            cv2.imshow('Result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Detect hammers in test images and draw bounding boxes
detect_hammers_and_draw_boxes(test_folder_path, clf)

