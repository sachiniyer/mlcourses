from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename, "rb") as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
            X = X.astype(np.float32) / 255.0
        with gzip.open(label_filename, "rb") as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            y = np.frombuffer(f.read(), dtype=np.uint8)
        self.images = X
        self.labels = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X, y = self.images[index], self.labels[index]
        if self.transforms:
            X = X.reshape((28, 28, -1))
            X = self.apply_transforms(X)
            return X.reshape(-1, 28 * 28), y
        return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION

    ### END YOUR SOLUTION
