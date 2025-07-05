import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color


def load_image(image_path):
    img = io.imread(image_path)
    img_gray = color.rgb2gray(img)
    return img_gray


def svd_compress(image, k):
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    compressed_image = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
    return compressed_image


def compression_ratio(image, k):
    original_size = image.size
    compressed_size = k * (image.shape[0] + image.shape[1] + 1)
    return compressed_size / original_size


def show_images(original_image, compressed_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Compressed Image')
    plt.imshow(compressed_image, cmap='gray')
    plt.axis('off')
    plt.show()


image_path = input('image_path:')
img = load_image(image_path)
k = int(input('k:'))
compressed_img = svd_compress(img, k)
show_images(img, compressed_img)
ratio = compression_ratio(img, k)
print(f"Compression ratio: {ratio:.2f}")
