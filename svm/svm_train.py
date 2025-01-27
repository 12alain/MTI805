import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse

def train_svm(embedding_path, svm_path, label_encoder_path):
    """Train an SVM model and save it."""
    print("[INFO] Loading embeddings...")
    with open(embedding_path, "rb") as f:
        data = pickle.load(f)

    print("[INFO] Encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    print("[INFO] Training SVM model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # Save the trained SVM model
    with open(svm_path, "wb") as f:
        pickle.dump(recognizer, f)
    print(f"[INFO] SVM model saved to {svm_path}")

    # Save the label encoder
    with open(label_encoder_path, "wb") as f:
        pickle.dump(le, f)
    print(f"[INFO] Label encoder saved to {label_encoder_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embeddings", required=False, default="output/embeddings.pickle",
                    help="Path to serialized db of facial embeddings")
    ap.add_argument("-s", "--svm", required=False, default="output/recognizer.pickle",
                    help="Path to output SVM model")
    ap.add_argument("-l", "--le", required=False, default="output/le.pickle",
                    help="Path to output label encoder")
    args = vars(ap.parse_args())

    train_svm(args["embeddings"], args["svm"], args["le"])
