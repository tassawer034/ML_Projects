# predict_image.py
import argparse
import joblib
from PIL import Image
import numpy as np

def load_and_prepare(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")   # grayscale
    img = img.resize((28, 28), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)

    # In many MNIST examples digits are dark on white background: invert so strokes ~1.0
    if arr.mean() > 127:
        arr = 255.0 - arr
    arr = arr / 255.0

    return arr.reshape(1, -1)

def main(args):
    bundle = joblib.load(args.model_path)
    model = bundle['model']
    scaler = bundle.get('scaler')
    pca = bundle.get('pca', None)

    x = load_and_prepare(args.image_path)
    if scaler is None:
        raise RuntimeError("Saved model bundle missing scaler.")

    # Apply scaler and PCA (if available)
    x_scaled = scaler.transform(x)
    if pca is not None:
        x_scaled = pca.transform(x_scaled)

    # Prediction
    pred = model.predict(x_scaled)
    print("Predicted digit:", int(pred[0]))

    # Safe probability handling
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(x_scaled)[0]
            # Fix: replace NaN or inf with 0
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

            # print top 3 probabilities
            top3 = sorted(enumerate(probs), key=lambda t: t[1], reverse=True)[:3]
            print("Top probabilities:")
            for cls, p in top3:
                print(f"  {cls}: {p:.4f}")
        except Exception as e:
            print("Probability calculation failed:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/logreg_mnist.joblib")
    parser.add_argument("--image_path", required=True)
    args = parser.parse_args()
    main(args)
