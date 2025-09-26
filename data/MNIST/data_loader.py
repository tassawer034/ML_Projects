# data_loader.py
import os
import struct
import numpy as np

def _read_idx_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols)
    return data

def _read_idx_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_mnist(data_dir: str = "./data"):
    """
    Find and load MNIST IDX files from data_dir.
    Accepts both filename variants (with or without hyphen).
    Returns: X_train, y_train, X_test, y_test (numpy arrays)
    """
    candidates = {
        'train_images': ['train-images.idx3-ubyte', 'train-images-idx3-ubyte'],
        'train_labels': ['train-labels.idx1-ubyte', 'train-labels-idx1-ubyte'],
        'test_images':  ['t10k-images.idx3-ubyte', 't10k-images-idx3-ubyte'],
        'test_labels':  ['t10k-labels.idx1-ubyte', 't10k-labels-idx1-ubyte'],
    }

    def find_file(choices):
        for c in choices:
            p = os.path.join(data_dir, c)
            if os.path.exists(p) and os.path.isfile(p):
                return p
        return None

    ti = find_file(candidates['train_images'])
    tl = find_file(candidates['train_labels'])
    xi = find_file(candidates['test_images'])
    xl = find_file(candidates['test_labels'])

    if not (ti and tl and xi and xl):
        missing = [k for k,v in [('train_images',ti), ('train_labels',tl), ('test_images',xi), ('test_labels',xl)] if v is None]
        raise FileNotFoundError(f"Missing MNIST files in {data_dir}. Missing: {missing}")

    X_train = _read_idx_images(ti)
    y_train = _read_idx_labels(tl)
    X_test  = _read_idx_images(xi)
    y_test  = _read_idx_labels(xl)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    print("Quick test of data loader (loading ./data)...")
    X_train, y_train, X_test, y_test = load_mnist("./data")
    print("Loaded shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
