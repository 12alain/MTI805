import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

def train_knn(embedding_path, knn_path, label_encoder_path, neighbors=5):
    """Train a KNN model and save it."""
    print("[INFO] Loading embeddings...")
    with open(embedding_path, "rb") as f:
        data = pickle.load(f)

    print("[INFO] Encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    with open(knn_path, "wb") as f:
        pickle.dump(recognizer, f)
    print(f"[INFO] KNN model saved to {knn_path}")

    with open(label_encoder_path, "wb") as f:
        pickle.dump(le, f)
    print(f"[INFO] Label encoder saved to {label_encoder_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embeddings", required=False, default="output/embeddings.pickle",
                    help="path to serialized db of facial embeddings")
    ap.add_argument("-k", "--knn", required=False, default="output/knn.pickle",
                    help="path to output KNN model")
    ap.add_argument("-l", "--le", required=False, default="output/le.pickle",
                    help="path to output label encoder")
    ap.add_argument("-n", "--neighbors", type=int, default=5,
                    help="number of neighbors for KNN")
    args = vars(ap.parse_args())

    train_knn(args["embeddings"], args["knn"], args["le"], args["neighbors"])
