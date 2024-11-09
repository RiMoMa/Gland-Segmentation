# ------------------------------------------------------------------------
# Pathology slide image processing script
# Creator: Ricardo Moncayo ralemom@gmail.com
# Date Created: some day in octuber 2024
#
# This script processes pathology slide images, extracting and
# saving regions of interest identified in the data from 'quality_data_coords.csv' file.
# Each region is evaluated and the one with the largest area for each slide is selected. The images of these
# selected regions are then saved. Basically extract annotated rectangles labeled as 1 2 3 4 5, if there are multiple choose the largest
# ------------------------------------------------------------------------

import ast
import os
from PIL import Image
import pandas as pd
import numpy as np
from openslide import open_slide
from concurrent.futures import ThreadPoolExecutor
Image.MAX_IMAGE_PIXELS = 642966272

def process_file(filename):
    directory = '/media/ricardo/Datos/SVS_named/corrected/'
    fileNameCsv = 'quality_data_coords.csv'
    #directory = '/GastricFinished/AllData/'
    #directoryOutput = '/media/ricardo/Datos/Regiones/'
    directoryOutput = '/media/ricardo/Datos/RegionesSVSNAMED/'
    file_path_CSV = directory + fileNameCsv

    quality_df = pd.read_csv(file_path_CSV)
    #Search in the dataframe where row corresponds to filename
    quality_df = quality_df[quality_df['Regions'].apply(lambda x: ast.literal_eval(x) != [])]
    # extract annotated regions coords
    AllSlideRegions = quality_df[quality_df['File Name'] == filename]['Regions']

    slide_path = os.path.join(directory, filename)
    slide = open_slide(slide_path)

    #a new DataFrame with columns named 'Region Name', 'x', 'y', 'w', 'h'
    reconstructed_list = ast.literal_eval(AllSlideRegions.iloc[0])
    df = pd.DataFrame(reconstructed_list, columns=['Region Name', 'x', 'y', 'w', 'h'])
    df['Region Name'] = np.uint(df['Region Name'])
    region_names = [1, 2, 3, 5,4]
    df['Area'] = df['w'] * df['h']

    for name in region_names:

        largest_region = df[df['Region Name'] == name].nlargest(1, 'Area')

        for idx, row in largest_region.iterrows():
            row['x'] = np.uint(row['x'])
            row['y'] = np.uint(row['y'])
            row['w'] = np.uint(row['w'])
            row['h'] = np.uint(row['h'])

            cropped_image_filename = "{}_X_{}_Y_{}_W_{}_H_{}_{}_{}.png".format(
                filename[:-4],
                int(row['x']),
                int(row['y']),
                int(row['w'] * 1),
                int(row['h'] * 1),
                'Region',
                int(row['Region Name']),
                idx if largest_region.shape[0] > 1 else ""
            )

            if not os.path.exists(directoryOutput + cropped_image_filename):
                print('processing: ', directoryOutput + cropped_image_filename)
                cropped_image = np.array(slide.read_region((np.uint(row['x']), np.uint(row['y'])), 0,
                                                           (np.uint(row['w'] * 16), np.uint(row['h'] * 16))))



                pil_image = (Image.fromarray(cropped_image))
                pil_image.save(directoryOutput + cropped_image_filename)
            else:
                print('exist: ',directoryOutput + cropped_image_filename)

   # print(reconstructed_list)
  #  print(type(reconstructed_list))


if __name__ == '__main__':
    directory = '/GastricFinished/AllData/'
    directory = '/media/ricardo/Datos/SVS_named/corrected/'

    file_names = os.listdir(directory)

    with ThreadPoolExecutor(max_workers=min(12, len(file_names))) as executor:
        executor.map(process_file, file_names)
