import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim

# Paths to grayscale and color images
grayscale_folder = "./gray"  # Path to the folder containing grayscale images
color_folder = "./color_256"  # Path to the folder containing original color images
output_folder = "./output"     # Path to the folder to save colorized images

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize PSNR and SSIM accumulators
total_psnr = 0
total_ssim = 0
num_images = 0  # Initialize the number of processed images

def calculate_psnr(original, colorized):
    """Calculate PSNR between original and colorized images."""
    mse = np.mean((original.astype(np.float32) - colorized.astype(np.float32)) ** 2)
    if mse == 0:
        return 100  # Infinite PSNR
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(original, colorized):
    """Calculate SSIM between original and colorized images."""
    # Determine the window size based on the smaller dimension of the images
    win_size = min(original.shape[:2])  # Use the smaller dimension for win_size
    if win_size < 7:
        return None  # Return None to skip SSIM calculation

    return ssim(original, colorized, multichannel=True, win_size=win_size)

def colorize_image(gray_image):
    """Colorize the grayscale image using the pre-trained model."""
    # Load the pre-trained model
    PROTOTXT = "./model/colorization_deploy_v2.prototxt"
    MODEL = "./model/colorization_release_v2.caffemodel"
    POINTS = "./model/pts_in_hull.npy"

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Convert the grayscale image to a 3-channel image
    gray_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Colorize image
    scaled = gray_3ch.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)  # Convert BGR to LAB

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (gray_image.shape[1], gray_image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    return (255 * colorized).astype("uint8")  # Convert to uint8

# Loop through all grayscale images
for filename in os.listdir(grayscale_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load grayscale image
        gray_image_path = os.path.join(grayscale_folder, filename)
        gray_image = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)

        # Load the corresponding color image
        color_image_path = os.path.join(color_folder, filename)
        color_image = cv2.imread(color_image_path)

        # Check if the color image exists
        if color_image is not None:
            # Colorize the grayscale image
            colorized_image = colorize_image(gray_image)

            # Save the colorized image for manual checking
            output_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_image_path, colorized_image)

            # Calculate PSNR
            psnr_value = calculate_psnr(color_image, colorized_image)
            total_psnr += psnr_value
            
            # Check if the image dimensions are suitable for SSIM calculation
            if gray_image.shape[0] >= 7 and gray_image.shape[1] >= 7:
                ssim_value = calculate_ssim(color_image, colorized_image)
                if ssim_value is not None:
                    total_ssim += ssim_value
                else:
                    print(f"Skipping SSIM calculation for {filename} due to insufficient window size.")
            else:
                print(f"Skipping SSIM calculation for {filename} due to small image size.")
            
            num_images += 1  # Increment the number of processed images
        else:
            print(f"Color image not found for {filename}")

# Calculate average PSNR and SSIM
if num_images > 0:
    average_psnr = total_psnr / num_images
    average_ssim = total_ssim / num_images if total_ssim > 0 else 0  # Avoid division by zero

    print(f"Average PSNR: {average_psnr}")
    print(f"Average SSIM: {average_ssim}")
else:
    print("No images processed.")
