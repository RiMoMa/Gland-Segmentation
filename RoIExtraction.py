import ast
from PIL import Image
import pandas as pd
import os
from openslide import open_slide
import numpy as np

#directory = '/GastricFinished/AllData/'
directory = '/media/ricardo/Datos/SVS_named/corrected/'
fileNameCsv = 'quality_data.csv'
file_path_CSV =directory+fileNameCsv

if fileNameCsv  in os.listdir():
    quality_df = pd.read_csv('quality_data_ROI.csv')

counter=0
for filename in os.listdir(directory):
    if filename.endswith('.svs'):
        AllSlideRegions = quality_df[quality_df['File Name'] == filename]['Regions']
        slide_path = os.path.join(directory, filename)
        slide = open_slide(slide_path)
    # Reconstruir la lista desde la cadena de caracteres
        reconstructed_list = ast.literal_eval(AllSlideRegions[0])

        df = pd.DataFrame(reconstructed_list, columns=['Region Name', 'x', 'y', 'w', 'h'])

        df['Region Name'] = np.uint(df['Region Name'])
        # Cargar la imagen original


    # Guardar la imagen recortada
        for idx, row in df.iterrows():
            # Recortar la imagen
            cropped_image = np.array(slide.read_region((row['x'], row['y']), 0, (row['w']*16, row['h']*16)))
            # Crear el nombre del archivo para la imagen recortada
            cropped_image_filename = "{}_X_{}_Y_{}_W_{}_H_{}_{}_{}.tiff".format(
                filename[:-4],  # Chops off the last 4 characters (.tif) from the filename
                row['x'],  # Uses the 'x' value from the row dictionary
                row['y'],  # Uses the 'y' value from the row dictionary
                row['w']*16,  # Uses the 'w' value from the row dictionary
                row['h']*16,  # Uses the 'h' value from the row dictionary
                row['Region Name'],  # Uses the 'Region Name' value from the row dictionary
                idx if df[df['Region Name'] == row['Region Name']].shape[0] > 1 else ""
                # Appends an index if there are more than one regions with the same name
            )

            # Guardar la imagen recortada
            pil_image = (Image.fromarray(cropped_image))
            pil_image.save(cropped_image_filename)

print(reconstructed_list)  # imprimir la lista reconstruida
print(type(reconstructed_list))