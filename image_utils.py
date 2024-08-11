import base64
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import cv2
import os
import replicate

# Constants for scaling
SCREENSHOT_DIMENSIONS = (2940, 1912)
CLICKING_DIMENSIONS = (1469, 955)
WIDTH_SCALING_FACTOR = CLICKING_DIMENSIONS[0] / SCREENSHOT_DIMENSIONS[0]
HEIGHT_SCALING_FACTOR = CLICKING_DIMENSIONS[1] / SCREENSHOT_DIMENSIONS[1]

# Constant for the model ID
SEGMENTATION_MODEL_ID = "lucataco/segment-anything-2:be7cbde9fdf0eecdc8b20ffec9dd0d1cfeace0832d4d0b58a071d993182e1be0"


def segment_image(base64_image, mask_limit=30):
    """Segments an image using a segmentation model on Replicate."""
    input = {
        "image": f"data:image/png;base64,{base64_image}",
        "mask_limit": mask_limit
    }

    output = replicate.run(
        SEGMENTATION_MODEL_ID,
        input=input
    )

    return output

def encode_image(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def crop_image_from_mask(image_path, mask_url, name_final_image='test'):
    """Crops an image based on a mask and returns the scaled centroid and encoded image."""
    # Load images from URLs
    response_mask = requests.get(mask_url)
    
    image = Image.open(image_path)
    mask = Image.open(BytesIO(response_mask.content))

    # Ensure mask is boolean
    mask = mask.convert("L")  # Convert mask to grayscale
    mask_array = np.array(mask)

    mask_array = mask_array > 128  # Convert to boolean where mask exists

    # Find contours
    contours, _ = cv2.findContours(mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the white mask
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the centroid of the contour
    M = cv2.moments(largest_contour)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    
    scaled_x = int(cX * WIDTH_SCALING_FACTOR)
    scaled_y = int(cY * HEIGHT_SCALING_FACTOR)

    # Apply mask to the image
    image_array = np.array(image)
    cropped_image_array = np.where(mask_array[:,:,None], image_array, 0)
    
    # Convert back to image
    cropped_image = Image.fromarray(cropped_image_array)
    
    # Find bounding box of the mask
    x_nonzero, y_nonzero = np.nonzero(mask_array)
    x_min, x_max = x_nonzero.min(), x_nonzero.max()
    y_min, y_max = y_nonzero.min(), y_nonzero.max()
    
    # Crop the image to the bounding box
    final_cropped_image = cropped_image.crop((y_min, x_min, y_max + 1, x_max + 1))
    
    # Create the folder if it does not exist
    os.makedirs("./images_crop", exist_ok=True)

    # Save the final image
    final_image_path = f"./images_crop/{name_final_image}.png"
    final_cropped_image.save(final_image_path)

    # encode the image
    return (scaled_x, scaled_y), encode_image(final_image_path)

if __name__ == "__main__":
    image_url = "screenshot.png"
    mask_url = "https://replicate.delivery/pbxt/Jaqg1OGIuuYIJJSUWGLeaVjbGgwNmFayLV4icgqe71gfqVeMB/mask_19.png"
    result = crop_image_from_mask(image_url, mask_url)
    print(result[0])
