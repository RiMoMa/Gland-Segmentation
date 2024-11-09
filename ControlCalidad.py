
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

# Crear una clase para proporcionar una caja de dialogo personalizada.
class CustomDialog(simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="Calidad del WSI:").grid(row=0)

        self.button_var = tk.StringVar()
        self.button_var.set("buena")  # set default value

        buena_button = tk.Radiobutton(master, text="Buena", variable=self.button_var, value="buena")
        buena_button.grid(row=1, column=0)

        artefactos_button = tk.Radiobutton(master, text="Con Artefactos", variable=self.button_var,
                                           value="con artefactos")
        artefactos_button.grid(row=2, column=0)

        borrosa_button = tk.Radiobutton(master, text="Borrosa", variable=self.button_var, value="borrosa")
        borrosa_button.grid(row=3, column=0)

        qupath_button = tk.Radiobutton(master, text="Qupath", variable=self.button_var, value="Qupath")
        qupath_button.grid(row=4, column=0)

        return buena_button  # initial focus

    def apply(self):
        self.result = self.button_var.get()



from pathlib import Path
import matplotlib.pyplot as plt
# DataFrame to store quality data
quality_df = pd.DataFrame(columns=['File Name', 'Quality'])

# Check whether data frame already exists and load it

directory = '/GastricFinished/AllData/'
directory ='/media/ricardo/Datos/SVS_named/corrected/'
fileNameCsv = 'quality_data.csv'
file_path_CSV =directory+fileNameCsv
if fileNameCsv in os.listdir(directory):
    quality_df = pd.read_csv(file_path_CSV)

# Directory where .svs files are stored

for filename in os.listdir(directory):
    if filename.endswith(".svs") and filename not in quality_df['File Name'].values:
        slide_path = os.path.join(directory, filename)
        slide = open_slide(slide_path)

        # Getting non-white regions (tissue regions)
        level = 3  # for half resolution

        np_slide = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))
        gray = cv2.cvtColor(np_slide, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((10,10), np.uint8)
        dilated = cv2.dilate(thresholded, kernel, iterations = 1)

        # Find contours of tissue regions
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Get the areas of all contours
        contour_areas = [cv2.contourArea(c) for c in contours]
        # Find the median of the largest elements
        median_area = np.mean(sorted(contour_areas,reverse=True))
        count = 0
        for c in contours:
            if count < 1:

                M = cv2.moments(c)
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if 0*np_slide.shape[1] <= cX <= np_slide.shape[1] and 0.1*np_slide.shape[0] <= cY <= 0.9*np_slide.shape[0]:
                    (x, y, w, h) = cv2.boundingRect(c)
                    ratio = float(h)/w
                    ratio  = math.degrees(math.atan(ratio))
                 #   print(ratio)
                    if ratio >15:
                        # If the area of the rectangle is greater than 50% of the median of the largest elements
                        area = cv2.contourArea(c)
                        if area > median_area:
                            np_slide = cv2.rectangle(np_slide, (x, y), (x + w, y + h), (0, 255, 0), 4)
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
                            thumbnail_resized = cv2.resize(thumbnail, (int(thumbnail.shape[1]*(region.shape[0]/thumbnail.shape[0])),region.shape[0]))
                            im_tile = np.concatenate((region, thumbnail_resized), axis=1)
                            cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
                            cv2.imshow(filename, im_tile[:, :, 2::-1])
                            cv2.resizeWindow(filename, int(w), int(h))

                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                            count = count + 1
            else:


                # Ask for Quality of WSI
                root = tk.Tk()
                root.withdraw()
                d = CustomDialog(root, "Calidad del WSI "+filename )
                print("result is", d.result)  # check the dialog result
                quality = d.result
                root.destroy()
                if quality == 'Qupath':
                    command = "/home/ricardo/Descargas/QuPath-v0.5.1-Linux/QuPath/bin/QuPath"
                    # Combine the command and the argument
                    full_command = [command, slide_path]

                    # Run the command
                    subprocess.run(full_command)

                    root = tk.Tk()
                    root.withdraw()
                    d = CustomDialog(root, "Calidad del WSI " + filename)
                    print("result is", d.result)  # check the dialog result
                    quality = d.result
                    root.destroy()
                    quality_df.loc[len(quality_df.index)] = {'File Name': filename, 'Quality': quality}
                    quality_df.to_csv(file_path_CSV, index=False)
                    print(quality_df.iloc[-1])
                    break

                else:
                # Add the quality and file name to the dataframe
                    quality_df.loc[len(quality_df.index)] = {'File Name': filename, 'Quality': quality}
                    quality_df.to_csv(file_path_CSV, index=False)
                    print(quality_df.iloc[-1])
                    break

                # Actions depending on the quality
                # if quality == 'buena':
                #     # Copy the .svs file to the Output folder
                #     shutil.copyfile(slide_path, os.path.join('Output', filename))
                # elif quality == 'con artefactos':
                #     # Save in OutputNoise
                #     shutil.copyfile(slide_path, os.path.join('OutputNoise', filename))
                # else:
                #     # Save in OutputBad
                #     shutil.copyfile(slide_path, os.path.join('OutputBad', filename))


# Saving the dataframe for future runs
