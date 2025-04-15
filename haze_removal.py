import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_dark_channel(image, window_size=15):
    """Calculate the dark channel of the image."""
    min_channel = np.amin(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def get_atmospheric_light(image, dark_channel):
    """Estimate the atmospheric light in the image."""
    flat_image = image.reshape(-1, 3)
    flat_dark = dark_channel.ravel()
    num_pixels = len(flat_dark)
    num_brightest = int(max(num_pixels * 0.001, 1))
    indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
    atmospheric_light = np.mean(flat_image[indices], axis=0)
    return atmospheric_light

def get_transmission_map(image, atmospheric_light, omega=0.95, window_size=15):
    """Estimate the transmission map of the image."""
    normalized_image = image / atmospheric_light
    transmission = 1 - omega * get_dark_channel(normalized_image, window_size)
    return transmission

def recover_image(image, transmission, atmospheric_light, t0=0.1):
    """Recover the scene radiance (dehazed image)."""
    transmission = np.maximum(transmission, t0)
    recovered_image = (image - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light
    recovered_image = np.clip(recovered_image, 0, 1)
    return recovered_image

def dehaze_image(image_path):
    """Dehaze an image using the Dark Channel Prior."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    dark_channel = get_dark_channel(image)
    atmospheric_light = get_atmospheric_light(image, dark_channel)
    transmission = get_transmission_map(image, atmospheric_light)
    dehazed_image = recover_image(image, transmission, atmospheric_light)

    # Plot the results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title('Dark Channel')
    plt.imshow(dark_channel, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title('Transmission Map')
    plt.imshow(transmission, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title('Dehazed Image')
    plt.imshow(dehazed_image)
    plt.axis('off')
    
    plt.show()

# Example usage
image_path = r"D:/Projects/Dehaze/hazy/29_hazy.png"
dehaze_image("D:/Projects/Dehaze/hazy")
