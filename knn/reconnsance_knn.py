import pickle
import cv2
import imutils
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import numpy as np
from imutils import paths
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


def recognize_faces_in_folder(folder_path, detector, embedder, recognizer, le, confidence_threshold):
    """Recognize faces in all images in a folder and evaluate with a confusion matrix."""
    image_paths = list(paths.list_images(folder_path))
    if not image_paths:
        print(f"[ERROR] No images found in the folder: {folder_path}")
        return

    print(f"[INFO] Found {len(image_paths)} images in the folder: {folder_path}")

    true_labels = []  # Étiquettes réellesy
    predicted_labels = []  # Étiquettes prédites

    # Parcourir chaque image et appliquer la reconnaissance faciale
    for image_path in image_paths:
        print(f"[INFO] Processing image: {image_path}")
        name = image_path.split(os.path.sep)[-2]  # Extraire la classe réelle depuis le chemin
        true_labels.append(name)  # Ajouter l'étiquette réelle

        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Unable to load image: {image_path}, skipping...")
            predicted_labels.append("Unknown")
            continue

        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        image_blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        detector.setInput(image_blob)
        detections = detector.forward()

        # Identifier le visage détecté avec la plus haute confiance
        face_detected = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                  (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(face_blob)
                vec = embedder.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                name_pred = le.classes_[j]
                predicted_labels.append(name_pred)  # Ajouter la prédiction
                face_detected = True
                break

        if not face_detected:
            predicted_labels.append("Unknown")  # Aucun visage détecté

    # Afficher la matrice de confusion et le rapport de classification
    print("[INFO] Generating confusion matrix...")
    cm = confusion_matrix(true_labels, predicted_labels, labels=le.classes_)
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=le.classes_))

    # Visualiser la matrice de confusion
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embeddings", required=False, default="output/embeddings.pickle",
                    help="path to serialized db of facial embeddings")
    ap.add_argument("-d", "--detector", required=False, default="face_detection_model",
                    help="path to OpenCV's deep learning face detector")
    ap.add_argument("-m", "--embedding-model", required=False, default="nn4.small2.v1.t7",
                    help="path to OpenCV's deep learning face embedding model")
    ap.add_argument("-r", "--recognizer", required=False, default="output/recognizer.pickle",
                    help="path to output model trained to recognize faces")
    ap.add_argument("-l", "--le", required=False, default="output/le.pickle",
                    help="path to output label encoder")
    ap.add_argument("-test", "--image", required=False, default="image",
                    help="path to the folder of test images for recognition")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # Charger les modèles
    print("[INFO] Loading models...")
    proto_path = os.path.sep.join([args["detector"], "deploy.prototxt"])
    model_path = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

    print("[INFO] Loading recognizer and label encoder...")
    with open(args["recognizer"], "rb") as f:
        recognizer = pickle.load(f)
    with open(args["le"], "rb") as f:
        le = pickle.load(f)

    # Tester toutes les images dans le dossier spécifié
    recognize_faces_in_folder(args["image"], detector, embedder, recognizer, le, args["confidence"])
