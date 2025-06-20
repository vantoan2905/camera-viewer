import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle

class Noise:
    def __init__(self, path_image):
        """
        Initializes the Noise class with a path to the image.
        :param path_image: str, path to the image file
        """
        self.path_image = path_image
        self.img = cv2.imread(self.path_image)

    def load_image(self):
        """Reloads the image from path."""
        self.img = cv2.imread(self.path_image)

    def save_image(self, img, filename):
        """Saves an image to disk."""
        cv2.imwrite(filename, img)

    def MedianFilter(self):
        """Applies a median filter to reduce salt-and-pepper noise."""
        return cv2.medianBlur(self.img, 5)

    def GaussianNoise(self):
        """
        Adds Gaussian noise to the image.
        """
        row, col, ch = self.img.shape
        mean = 0
        sigma = 25
        gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)
        noisy = self.img.astype(np.float32) + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def AdaptiveThreshold(self):
        """
        Applies adaptive Gaussian thresholding to the grayscale image.
        """
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    def ContraharmonicMeanFilter(self, size=3, q=1.5):
        """
        Applies contraharmonic mean filter to reduce noise, especially salt or pepper noise.
        """
        img = self.img.astype(np.float32)
        num = cv2.boxFilter(np.power(img, q + 1), ddepth=-1, ksize=(size, size))
        denom = cv2.boxFilter(np.power(img, q), ddepth=-1, ksize=(size, size))
        result = num / (denom + 1e-8)
        return np.clip(result, 0, 255).astype(np.uint8)

    def DecisionBasedSwitchingFilter(self):
        """
        Applies a decision-based switching filter to remove salt-and-pepper noise.
        """
        median = cv2.medianBlur(self.img, 3)
        diff = cv2.absdiff(self.img, median)
        mask = np.any(diff > 30, axis=2)  # 2D mask for 3-channel image
        result = self.img.copy()
        result[mask] = median[mask]
        return result

    def TotalVariationFilter(self):
        """
        Applies total variation denoising using the Chambolle algorithm.
        """
        img = self.img.astype(np.float32) / 255.0
        denoised = denoise_tv_chambolle(img, weight=0.1, channel_axis=-1)
        return (denoised * 255).astype(np.uint8)
