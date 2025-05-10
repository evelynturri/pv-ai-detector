import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def colorize_image(input_path, output_path):
    # Load the grayscale image
    image = Image.open(f'{input_path}').convert('L')  # 'L' mode ensures it's grayscale
    image_array = np.array(image)

    # Normalize the grayscale values to range [0, 1]
    normalized_array = image_array / 255.0

    # Define the gradient colors for red and blue
    # Red at grayscale 0, Blue at grayscale 255
    red = np.array([255, 0, 0])
    blue = np.array([0, 0, 255])

    # Create the color scale
    # Interpolating between red and blue
    color_mapped_array = (1 - normalized_array)[:, :, None] * blue + normalized_array[:, :, None] * red

    # Ensure values are valid for image (0-255) and convert to uint8
    color_mapped_array = np.uint8(color_mapped_array)

    # Create and save the colorized image
    colorized_image = Image.fromarray(color_mapped_array)
    colorized_image.save(f'{output_path}')

def apply_jet_colormap(input_path, output_path):
    # Load the grayscale image
    # Load the grayscale image
    image = Image.open(input_path).convert('L')  # Ensure it's grayscale
    image_array = np.array(image)
    
    # Normalize the grayscale values to range [0, 1]
    normalized_array = image_array / 255.0

    # Apply the 'jet' colormap from Matplotlib
    colormap = plt.cm.get_cmap('jet')  # Get the jet colormap
    color_mapped_array = colormap(normalized_array)  # Apply colormap

    # Remove the alpha channel and convert to uint8
    color_mapped_array = (color_mapped_array[:, :, :3] * 255).astype(np.uint8)  # Drop alpha and scale to [0, 255]

    # Create and save the colorized image
    colorized_image = Image.fromarray(color_mapped_array)
    colorized_image.save(output_path)

# Function to process all images in a folder
def process_folder(input_folder, output_folder, jet):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):  # Add other image formats if needed
            # Construct full file paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Apply the colorization to each image
            print(f'Processing {filename}...')
            if not jet:
                colorize_image(input_path, output_path)
            else:
                apply_jet_colormap(input_path, output_path)

    print('All images processed.')

# Colors
grey_scale_folder = 'dataset/InfraredSolarModules/images'
color_scale_folder = 'dataset/InfraredSolarModules_Color'

if not os.path.exists(color_scale_folder):
    os.makedirs(color_scale_folder)

color_scale_folder = f'{color_scale_folder}/images'

process_folder(grey_scale_folder, color_scale_folder, jet=False)




