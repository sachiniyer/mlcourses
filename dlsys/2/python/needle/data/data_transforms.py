import numpy as np


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            img = np.fliplr(img)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        shift_x, shift_y = np.random.randint(-self.padding, self.padding + 1, 2)
        ### BEGIN YOUR SOLUTION
        h, w, c = img.shape
        padded_img = np.pad(
            img,
            ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            "constant",
        )
        return padded_img[
            self.padding + shift_x : self.padding + shift_x + h,
            self.padding + shift_y : self.padding + shift_y + w,
            :,
        ]
        ### END YOUR SOLUTION
