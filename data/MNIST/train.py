# train.py
import argparse
import os
import joblib
from data_loader import load_mnist
from preprocess import prepare_data
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def build_and_train(args):
    print("Loading MNIST data from:", args.data_dir)
    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    print("Raw shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    if args.subset and args.subset > 0:
        print(f"Using subset of training data: {args.subset} samples")
    print("Preprocessing: normalize -> scaler -> (optional PCA)")
    scaler, pca, X_train_scaled, X_test_scaled, y_train_subset = prepare_data(X_train, X_test, y_train, subset=args.subset, do_pca=args.pca)

    # choose model
    if args.model == "logreg":
        print("Building LogisticRegression (multinomial, saga). This may take a while on the full dataset.")
        clf = LogisticRegression(
            penalty='l2',
            solver='saga',
            multi_class='multinomial',
            max_iter=args.max_iter,
            n_jobs=args.n_jobs,
            verbose=1,
            random_state=42
        )
    else:
        print("Building SGDClassifier (fast stochastic logistic)")
        clf = SGDClassifier(loss='log_loss', penalty='l2', max_iter=args.max_iter, tol=1e-3, random_state=42)

    try:
        print("Training started...")
        clf.fit(X_train_scaled, y_train_subset)
        print("Training finished.")
    except KeyboardInterrupt:
        print("Training interrupted by user (KeyboardInterrupt). Will try to save partial model if available.")
    except Exception as e:
        print("Training error:", e)
        raise

    # evaluate on test
    print("Evaluating on test set...")
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    # save artifacts
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    print(f"Saving model bundle to {args.save_path} ...")
    joblib.dump({'model': clf, 'scaler': scaler, 'pca': pca, 'args': vars(args)}, args.save_path)
    print("Saved.")

def parse_args():
    p = argparse.ArgumentParser(description="Train logistic regression on MNIST")
    p.add_argument("--data_dir", default="./data", help="Directory with MNIST idx files")
    p.add_argument("--save_path", default="models/logreg_mnist.joblib", help="Where to save model bundle")
    p.add_argument("--model", choices=["logreg", "sgd"], default="sgd", help="Which estimator to use (sgd is faster for quick runs)")
    p.add_argument("--subset", type=int, default=0, help="Use a training subset for quick runs (0 = full)")
    p.add_argument("--pca", type=int, default=0, help="If >0, reduce to this many PCA components")
    p.add_argument("--max_iter", type=int, default=200, help="Max iterations for estimator")
    p.add_argument("--n_jobs", type=int, default=1, help="n_jobs for LogisticRegression (saga)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_and_train(args)
