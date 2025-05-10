import os
import json
import shutil

def process_folder(source:str, destination:str):

    # Opening JSON file
    j = open('dataset/InfraredSolarModules/module_metadata.json')

    # returns JSON object as a dictionary
    dataset = json.load(j)
    j.close()

    # Loop through all files in the input folder
    for filename in os.listdir(source):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):  # Add other image formats if needed
            # Construct full file paths
            source_file = os.path.join(source, filename)
            
            file_class = dataset[f'{filename[:-4]}']['anomaly_class']

            destination_class = f'{destination}/{file_class}'
            if not os.path.exists(destination_class):
                os.makedirs(destination_class)

            # Apply the colorization to each image
            print(f'Processing {filename}...')
            shutil.copy(source_file, destination_class)

    print('All images processed.')
    return

# GreyScale Dataset
# source = 'dataset/InfraredSolarModules'
# destination = f'{source}_Classes'

# Color Dataset
source = 'dataset/InfraredSolarModules_JetColorMap'
destination = f'{source}_Classes'

if not os.path.exists(destination):
    os.makedirs(destination)

source = f'{source}/images'
destination = f'{destination}'

process_folder(source, destination)




