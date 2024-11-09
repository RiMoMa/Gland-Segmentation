"""
Object Detection & Interactive Labeling in Microscopic Tissue Scans

Author:Ricardo Moncayo
Date: some day in a phd project

This script looks for '.svs' files in a given directory and applies image
processing techniques to each found slide. It identifies unique tissue contours in
the slide and presents these to a user, allowing the user to interactively
label different regions of interest in each slide.

Any regions named by the user, along with their corresponding slides, are
recorded to a CSV file for potential later use (like training machine learning
models for cell or tissue recognition).

Requires OpenSlide for reading '.svs' files and OpenCV, PIL, NumPy and
scipy libraries for image processing and object detection tasks.
"""

import os
import cv2
import pandas as pd
import shutil
import numpy as np
from openslide import open_slide
from tkinter import messagebox, Tk
import math
import subprocess
import tkinter as tk
from tkinter import simpledialog
from skimage import measure, morphology
#import pyautogui
from pathlib import Path
import matplotlib.pyplot as plt
# DataFrame to store quality data
quality_df = pd.DataFrame(columns=['File Name', 'Regions'])


#directory = '/GastricFinished/AllData/'

directory = '/media/ricardo/Datos/SVS_named/corrected/'
fileNameCsv = 'quality_data_coords.csv'
file_path_CSV =directory+fileNameCsv

# Check whether data frame already exists and load it

if fileNameCsv in os.listdir(directory ):
    quality_df = pd.read_csv(file_path_CSV)

# Directory where .svs files are stored

counter=0
for filename in os.listdir(directory):
    counter =counter+1
    print('realizando ',counter)
    print('total ', len(os.listdir(directory)))
    if filename.endswith(".svs") and filename not in quality_df['File Name'].values:
        print('Evaluating ',filename)
        slide_path = os.path.join(directory, filename)
        slide = open_slide(slide_path)

        # Getting non-white regions (tissue regions)
        level = 3  # for half resolution

        np_slide = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))
        gray = cv2.cvtColor(np_slide, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
        min_pixel_count_threshold = 4100

        binary_mask = morphology.remove_small_objects(measure.label(thresholded), min_size=min_pixel_count_threshold)
        kernel = np.ones((10,10), np.uint8)
        #dilated = cv2.dilate(thresholded, kernel, iterations = 1)
        dilated = cv2.dilate(np.uint8(binary_mask > 0), kernel, iterations=1)
        # Etiquetar todos los grupos conectados en la imagen binaria
        labels = measure.label(dilated)
        # Remover grupos que tienen menos de un cierto número de píxeles
        binary_mask = morphology.remove_small_objects(labels, min_size=min_pixel_count_threshold)
        binary_image_4d = np.dstack([binary_mask>0] * 4)
        np_slide = np_slide * binary_image_4d
        contours, _ = cv2.findContours(np.array((binary_mask>0)*1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(np.array(binary_mask).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rgb_image = np_slide
        coordinates = []

        for i, contour in enumerate(contours):
            # Obtener el bounding box del contorno
            x, y, w, h = cv2.boundingRect(contour)

            # Extraer la región de interés de la imagen RGB
            roi_rgb = rgb_image[y:y + h, x:x + w]

            # Comprobar la intersección
            if np.any(binary_mask[y:y + h, x:x + w]>0 & (roi_rgb[:, :, 0] > 0)):
                # Mostrar la región detectada al usuario
                np_slideCopy = rgb_image.copy()
                np_slide = cv2.rectangle(np_slideCopy, (x, y), (x + w, y + h), (0, 255, 0), 4)
                # cv2.namedWindow('Quality Control', cv2.WINDOW_NORMAL)
                # cv2.imshow('Quality Control', np_slide[:,:,2::-1])
                # cv2.resizeWindow('Quality Control', int(np_slide.shape[1] / 3), int(np_slide.shape[0] / 3))
                upsample = slide.level_downsamples[level]
                x = int(x * upsample)
                y = int(y * upsample)
                w = int(w * upsample / slide.level_downsamples[level - 1])
                h = int(h * upsample / slide.level_downsamples[level - 1])

                thumbnail = np_slide
                region = np.array(slide.read_region((x + w, y + h), level - 3, (w * 4, h * 4)))
                thumbnail_resized = cv2.resize(thumbnail, (
                int(thumbnail.shape[1] * (region.shape[0] / thumbnail.shape[0])), region.shape[0]))
                im_tile = np.concatenate((region, thumbnail_resized), axis=1)

                cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
                cv2.imshow(filename, thumbnail_resized[:, :, 2::-1])
                cv2.resizeWindow(filename, int(w), int(h))

                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Pedir al usuario que nombre la región o la descarte
                region_name = input(f"Nombre para la región {i + 1} (o presiona 'd' para descartar): ").strip()

                if region_name.lower() != 'd':
                    coordinates.append((region_name, x, y, w, h))
        quality_df.loc[len(quality_df.index)] = {'File Name': filename, 'Regions': coordinates}
        quality_df.to_csv(file_path_CSV, index=False)
        print(quality_df.iloc[-1])

    else:
        print('Exist ',filename)# Cerrar la ventana de visualización

