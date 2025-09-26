# evaluate.py
import argparse
import joblib
from data_loader import load_mnist
from preprocess import normalize_pixels
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import numpy as np

def main(args):
    print("Loading model:", args.model_path)
    bundle = joblib.load(args.model_path)
    model = bundle['model']
    scaler = bundle.get('scaler', None)
    pca = bundle.get('pca', None)

    print("Loading MNIST test set from:", args.data_dir)
    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    X_test = normalize_pixels(X_test)

    if scaler is None:
        raise RuntimeError("Saved model bundle does not contain a scaler.")
    X_test_scaled = scaler.transform(X_test)
    if pca is not None:
        X_test_scaled = pca.transform(X_test_scaled)

    print("Predicting...")
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    print("Classification report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    out_png = os.path.splitext(args.model_path)[0] + "_confusion.png"
    plt.tight_layout()
    plt.savefig(out_png)
    print("Saved confusion matrix to", out_png)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--model_path", default="models/logreg_mnist.joblib")
    args = parser.parse_args()
    main(args)
