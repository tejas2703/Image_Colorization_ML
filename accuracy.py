import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Paths to the original and colorized images
ORIGINAL_FOLDER = "./color_256"  # Path to the folder containing original color images
COLORIZED_FOLDER = "./output"  # Path to the folder containing colorized images

# Function to calculate PSNR
def calculate_psnr(original, colorized):
    mse = np.mean((original - colorized) ** 2)
    if mse == 0:  # if images are identical
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

# Function to calculate SSIM
def calculate_ssim(original, colorized):
    return ssim(original, colorized, multichannel=True)

# Initialize metrics
psnr_values = []
ssim_values = []

# Get the list of images
original_images = sorted(os.listdir(ORIGINAL_FOLDER))
colorized_images = sorted(os.listdir(COLORIZED_FOLDER))

# Ensure both folders have the same number of images
if len(original_images) != len(colorized_images):
    raise ValueError("The number of original images and colorized images must be the same.")

# Calculate PSNR and SSIM for each pair of images
for original_filename, colorized_filename in zip(original_images, colorized_images):
    # Load the original and colorized images
    original_path = os.path.join(ORIGINAL_FOLDER, original_filename)
    colorized_path = os.path.join(COLORIZED_FOLDER, colorized_filename)
    
    original_image = cv2.imread(original_path)
    colorized_image = cv2.imread(colorized_path)

    # Calculate PSNR
    psnr_value = calculate_psnr(original_image, colorized_image)
    psnr_values.append(psnr_value)

    # Calculate SSIM
    ssim_value = calculate_ssim(original_image, colorized_image)
    ssim_values.append(ssim_value)

# Calculate average metrics
average_psnr = np.mean(psnr_values)
average_ssim = np.mean(ssim_values)

print(f"Average PSNR: {average_psnr:.2f}")
print(f"Average SSIM: {average_ssim:.4f}")
