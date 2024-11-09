import numpy as np
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
import openslide
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
import subprocess
def process_all_images(path_folder, patchsize, batch_size, HighResol, ImgNormalize):
    # Listar todos los archivos en la carpeta
    all_files = os.listdir(path_folder)
    # Filtrar solo archivos svs
    svs_files = [file for file in all_files if file.endswith('.svs')]
    # Procesar cada archivo
    for svs_file in svs_files:
        path_svs = os.path.join(path_folder, svs_file)
        outputIm = svs_file.replace('.svs', '.jpc')
        # La última parte toma en cuenta solo el nombre del archivo
        case_name = os.path.basename(svs_file).replace('.svs', '')
        print('extracting: ',case_name)
        generate_Patches(patchsize, batch_size, path_svs, outputIm=outputIm, HighResol=HighResol, ImgNormalize=ImgNormalize,CaseName=case_name)


from sklearn import preprocessing


def save_patient(imgSVS, x, y, patchsize, name, PathDataset):
    ImgPathSave = PathDataset + 'img_X_' + str(x) + '_Y_' + str(y) + '_' + str(name[0]) + '_' + str(
        name[1]) + '.png'

    if not os.path.exists(ImgPathSave):
        img = imgSVS.read_region((x, y), 0, (patchsize, patchsize))
        img = np.array(img)[:, :, 0:3]
        # Condición de suma de blanco
        is_white = np.sum(img.ravel() > 220) / len(img.ravel()) > 0.9

        if not is_white:
            img = Image.fromarray(img)
            # save a
            img.save(ImgPathSave)

    # img = img.resize((512, 512),Image.BILINEAR)
    return 0

import openslide
import os
from concurrent.futures import ThreadPoolExecutor

import openslide
import os
from concurrent.futures import ThreadPoolExecutor

def generate_Patches(patchsize=(512,512), batch_size=100, path_svs = None,outputIm=None,HighResol=False, allGpus=False,ImgNormalize=False,CaseName=None,OutputPath=None):
    # open svs
    slide = openslide.OpenSlide(path_svs)
    svs_size = slide.level_dimensions[0]
    # Calculate the size of an image in float32 format (assuming each pixel in the image is a float32).

    num_rows = svs_size[1] // patchsize[1]
    num_columns = svs_size[0] // patchsize[0]
    extraction_coords = {(j, i): (i * patchsize[0], j * patchsize[1]) for i in range(num_rows) for j in range(num_columns)} # enviar las cordenadas al dataloader personalizado
    extraction_coords = {(i, j): (i * patchsize[0], j * patchsize[1]) for i in range(num_rows) for j in
                         range(num_columns)}  # enviar las cordenadas al dataloader personalizado
    extraction_coords = {(j, i): (j * patchsize[0], i * patchsize[1]) for i in range(num_rows) for j in
                         range(num_columns)}  # enviar las cordenadas al dataloader personalizado
    gland_img = [(k, v) for k, v in extraction_coords.items()]

    PathDataset = OutputPath+'/' + CaseName + '/'
    os.makedirs(PathDataset, exist_ok=True)
    MAX_WORKERS = 12
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, (nombre, (x, y)) in enumerate(gland_img):
            executor.submit(save_patient, slide, x, y, patchsize[0], nombre, PathDataset)