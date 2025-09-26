"""
train_mnist_sklearn.py

Train / evaluate a sklearn MNIST classifier and optionally predict on image files.

Usage examples:
  # Train (logistic regression)
  python train_mnist_sklearn.py --model-out models/mnist_model.joblib --model-type logreg --max-iter 1000 --retrain

  # Train (MLP)
  python train_mnist_sklearn.py --model-out models/mnist_mlp.joblib --model-type mlp --max-iter 200 --retrain

  # Predict on custom images (after training / or with --train-if-missing)
  python train_mnist_sklearn.py --model-out models/mnist_model.joblib --predict samples/digit1.png samples/digit2.png

This script automatically:
- loads MNIST (keras), flattens and normalizes (0..1)
- trains a scikit-learn LogisticRegression or MLPClassifier
- saves the model with joblib
- evaluates on MNIST test set and prints a classification report
- preprocesses input images (28x28, grayscale) and auto-inverts if needed
"""

import os
import argparse
import numpy as np
import joblib
import warnings
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2


def preprocess_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # flatten + normalize
    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
    x_test  = x_test.reshape(-1, 28*28).astype('float32') / 255.0
    return x_train, y_train, x_test, y_test


def preprocess_image_file(path):
    """Read an image file, convert to 28x28 grayscale, normalize and flatten.
       Auto-invert if mean pixel value suggests background is white.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Can't read image: {path}")
    # resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    mean_val = img.mean()
    # MNIST digits have white(1) ink on black(0) background; sometimes custom images are inverted
    if mean_val > 0.5:
        # likely white background with dark digit -> invert
        img = 1.0 - img
    # Flatten to (1,784)
    return img.reshape(1, -1)


def build_model(model_type, max_iter):
    if model_type == 'logreg':
        # saga works well for large datasets and multinomial logistic
        model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=max_iter, n_jobs=-1, verbose=1)
    else:
        # small MLP; change hidden_layer_sizes if you want a bigger model
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=max_iter, verbose=True, random_state=42)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-out', default='models/mnist_model.joblib', help='where to save/load model')
    p.add_argument('--model-type', choices=['logreg', 'mlp'], default='logreg')
    p.add_argument('--max-iter', type=int, default=1000)
    p.add_argument('--retrain', action='store_true', help='force retraining even if model file exists')
    p.add_argument('--train-if-missing', action='store_true', help='train automatically if model file is missing when predicting')
    p.add_argument('--predict', nargs='*', help='image file(s) to predict (PNG/JPG)')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)

    # reduce noisy sklearn convergence warning clutter (we still print them if useful)
    warnings.filterwarnings('default', category=UserWarning)

    x_train, y_train, x_test, y_test = preprocess_dataset()

    model = None
    if os.path.exists(args.model_out) and not args.retrain:
        try:
            print(f"Loading model from {args.model_out} ...")
            model = joblib.load(args.model_out)
        except Exception as e:
            print("Failed to load model — will retrain. Error:", e)
            model = None

    if (model is None) and (args.predict and not args.train_if_missing):
        print("Model not found. Use --train-if-missing to auto-train, or run with --retrain to force training.")

    if model is None:
        print('Training model (this may take a while)...')
        model = build_model(args.model_type, args.max_iter)
        # fit
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(x_train, y_train)
        joblib.dump(model, args.model_out)
        print('Saved model to', args.model_out)

    # Evaluate on test set
    print('\nEvaluating on MNIST test set...')
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test accuracy: {acc:.4f}')
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, digits=4))

    # optional predictions on provided images
    if args.predict:
        for pth in args.predict:
            try:
                x = preprocess_image_file(pth)
            except Exception as e:
                print(f"Skipping {pth}: {e}")
                continue

            # get probs if available
            probs = None
            if hasattr(model, 'predict_proba'):
                try:
                    probs = model.predict_proba(x)[0]
                except Exception as e:
                    print(f"Warning: predict_proba failed for {pth}: {e}")
            pred = model.predict(x)[0]

            print('\nFile:', pth)
            print(' Predicted digit:', int(pred))
            if probs is not None:
                # show top-3 probabilities
                top_idx = np.argsort(probs)[::-1][:3]
                print(' Top probabilities:')
                for rank, idx in enumerate(top_idx):
                    print(f'  {rank+1}: {idx}: {probs[idx]:.4f}')


if __name__ == '__main__':
    main()
