from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
import argparse


def load_models(detector_path, embedding_model_path):
    """Load the face detector and embedding model."""
    print("[INFO] Loading face detector...")
    proto_path = os.path.sep.join([detector_path, "deploy.prototxt"])
    model_path = os.path.sep.join([detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    print("[INFO] Loading face embedding model...")
    embedder = cv2.dnn.readNetFromTorch(embedding_model_path)
    return detector, embedder


def create_embeddings(image_paths, detector, embedder, confidence_threshold, save_path):
    """Process images to extract embeddings and names."""
    print("[INFO] Creating embeddings...")
    known_embeddings = []
    known_names = []
    total = 0

    for (i, imagePath) in enumerate(image_paths):
        print(f"[INFO] Processing image {i + 1}/{len(image_paths)}")
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)

        if image is None:
            print(f"[WARNING] Unable to load {imagePath}")
            continue

        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        image_blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        detector.setInput(image_blob)
        detections = detector.forward()

        for j in range(detections.shape[2]):
            confidence = detections[0, 0, j, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                  (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(face_blob)
                vec = embedder.forward()

                known_names.append(name)
                known_embeddings.append(vec.flatten())
                total += 1

    print(f"[INFO] Serialized {total} encodings...")
    data = {"embeddings": known_embeddings, "names": known_names}
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] Embeddings saved to {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", required=False, default="dataset",
                    help="path to input directory of faces + images")
    ap.add_argument("-e", "--embeddings", required=False, default="output/embeddings.pickle",
                    help="path to output serialized db of facial embeddings")
    ap.add_argument("-d", "--detector", required=False, default="face_detection_model",
                    help="path to OpenCV's deep learning face detector")
    ap.add_argument("-m", "--embedding-model", required=False, default="nn4.small2.v1.t7",
                    help="path to OpenCV's deep learning face embedding model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # Load models
    detector, embedder = load_models(args["detector"], args["embedding_model"])

    # Get image paths
    image_paths = list(paths.list_images(args["dataset"]))

    # Create embeddings
    create_embeddings(image_paths, detector, embedder, args["confidence"], args["embeddings"])
