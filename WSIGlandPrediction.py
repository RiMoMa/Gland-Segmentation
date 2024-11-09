import numpy as np
from PIL import Image
import openslide
import utils
import torch
import math
from tqdm import tqdm
################################
### OPEN  Dataframe ##################
import pandas as pd
import re

def preprocess_data(file_path):
    # Leer el archivo csv y guardar los datos en un DataFrame
    MetaplasiaPreds = pd.read_csv(file_path)

    # Definir una función para extraer las coordenadas de los nombres de las imágenes
    def extract_coordinates(image_name):
        # Buscar las coordenadas x en el nombre de la imagen y guardar el resultado
        match = re.search(r'_x(\d+)_', image_name)
        # Si se encuentra una coincidencia, convertir el resultado en un entero. Si no, devolver None
        x_coord = int(match.group(1)) if match else None

        # Buscar las coordenadas y en el nombre de la imagen y guardar el resultado
        match = re.search(r'_y(\d+)[^\d]', image_name)
        # Si se encuentra una coincidencia, convertir el resultado en un entero. Si no, devolver None
        y_coord = int(match.group(1)) if match else None

        # Devolver las coordenadas x e y como una tupla
        return x_coord, y_coord

    # Añadir una nueva columna al DataFrame con los nombres de las imágenes extraídos de la columna 'name'
    MetaplasiaPreds['image_name'] = MetaplasiaPreds['name'].str.split('/').str[-1]

    # Añadir una nueva columna al DataFrame con los nombres de los casos extraídos de la columna 'image_name'
    MetaplasiaPreds['case_name'] = MetaplasiaPreds['image_name'].str.split('_').str[0]

    # Aplicar la función extract_coordinates a la columna 'image_name' y dividir el resultado en dos nuevas columnas
    MetaplasiaPreds['x_coord'], MetaplasiaPreds['y_coord'] = zip(
        *MetaplasiaPreds['image_name'].apply(extract_coordinates))

    # Convertir las columnas 'x_coord' y 'y_coord' en enteros
    MetaplasiaPreds['x_coord'] = MetaplasiaPreds['x_coord'].astype(int)
    MetaplasiaPreds['y_coord'] = MetaplasiaPreds['y_coord'].astype(int)

    # Cambiar el índice del DataFrame a la columna 'case_name'
    MetaplasiaPreds.set_index('case_name', inplace=True)

    # Devolver el DataFrame
    return MetaplasiaPreds


# Uso de la función
file_path = '/home/ricardo/Documentos/IsbiAutomatico/Dataset/convnext.data.csv'
MetaplasiaPreds = preprocess_data(file_path)

######################################3
#####################################

####Function for pred metaplasia
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops


def process_glands(MetaplasiaPredsCase, res, x, y):
    glands = np.array(res)[:, :, 0] > 0
    labels = label(glands)
    bbox_props = regionprops(labels)
    canvas = np.zeros_like(glands).astype(np.uint8) * 1
    mask = np.zeros_like(glands).astype(np.uint8)

    #parche grande
    minr_img =  y
    minc_img =  x
    maxr_img = y+np.shape(canvas)[1]
    maxc_img =  x+np.shape(canvas)[0]

    intersection_x =(((MetaplasiaPredsCase['x_coord'] >= minc_img) &
                     (MetaplasiaPredsCase['x_coord'] <= maxc_img)) |
                     ((MetaplasiaPredsCase['x_coord']+512 >= minc_img) & (MetaplasiaPredsCase['x_coord'] <= maxc_img)))
    intersection_y = (((MetaplasiaPredsCase['y_coord'] >= minr_img) &
                      (MetaplasiaPredsCase['y_coord'] <= maxr_img)) |
                      (
                (MetaplasiaPredsCase['y_coord'] + 512 >= minr_img) & (MetaplasiaPredsCase['y_coord'] <= maxr_img))
                      )
    intersection_all = intersection_x & intersection_y
    MetaplasiaPredsCasePatch = MetaplasiaPredsCase[intersection_all]

    for index, row in MetaplasiaPredsCasePatch.iterrows():
        pred_box = (row['x_coord'], row['y_coord'], row['x_coord'] + 256 * 2, row['y_coord'] + 256 * 2)
        Glandlabel = row['pred'] + 1
        canvas[pred_box[1] - y:pred_box[3] - y, pred_box[0] - x:pred_box[2] - x] = Glandlabel



    for prop_num in enumerate(bbox_props):
        prop = bbox_props[prop_num[0]]

        coordinates = prop.coords
        GlandType = []
        for (pa,pb) in coordinates:
            GlandType.append(canvas[pa,pb])
        counts = np.bincount(GlandType)
        most_common = np.argmax(counts)
        for (pa,pb) in coordinates:
            mask[pa,pb] = most_common

    return labels, mask


def save_npy(data, file_name):
    np.save(file_name, data)


def overlay(canvas, imgRgb):
    # Assuming canvas and imgRgb are the same shape and dtype...
    overlay_image = np.dstack((canvas, imgRgb))
    return overlay_image
    #plt.imshow(overlay_image)
    #plt.show()


# Uso de las funciones
#labels, canvas = process_glands(MetaplasiaPredsCase, res, x, y)
#save_npy(labels, 'labels.npy')
#save_npy(canvas, 'canvas.npy')
#overlay(canvas, imgRgb)


import os
def generate_WSI(patchsize=(512,512), batch_size=100, path_svs = None,outputIm=None,HighResol=False, allGpus=False,ImgNormalize=False,MetaplasiaPredsCase=None,CaseName=None):
    # open svs
    slide = openslide.OpenSlide(path_svs)
    svs_size = slide.level_dimensions[0]
    # Calculate the size of an image in float32 format (assuming each pixel in the image is a float32).

    float32_image_size = patchsize[0] * patchsize[1] * 4 * 3  # 4 bytes for float32
    import psutil
    # Calculate the RAM available for storing images
    available_ram = psutil.virtual_memory().available

    # Calculate the max number of images that can fit in the RAM.
    max_images_in_ram = available_ram // float32_image_size

    # Determine if max_images_in_ram is greater than batchsize
    if max_images_in_ram > batch_size:
        print("Max number of images that can be stored in RAM is greater than the batch size.")
    else:
        print("Max number of images that can be stored in RAM is not greater than the batch size.")


    num_rows = svs_size[1] // patchsize[1]
    num_columns = svs_size[0] // patchsize[0]
    extraction_coords = {(j, i): (i * patchsize[0], j * patchsize[1]) for i in range(num_rows) for j in range(num_columns)} # enviar las cordenadas al dataloader personalizado
    extraction_coords = {(i, j): (i * patchsize[0], j * patchsize[1]) for i in range(num_rows) for j in
                         range(num_columns)}  # enviar las cordenadas al dataloader personalizado
    extraction_coords = {(j, i): (j * patchsize[0], i * patchsize[1]) for i in range(num_rows) for j in
                         range(num_columns)}  # enviar las cordenadas al dataloader personalizado

    #hacer el forde las predicciones del modelo generar las predicciones y luego con kakadu generar el WSI
    num_batches = math.ceil(len(extraction_coords) / batch_size)
    import scipy.io

    ##### Cargar MODELO UNET #####
    import warnings
    import dataloaderSVSGlands as dataloader
    from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
    import segmentation_models_pytorch as smp
    from torchsummary import summary
    import subprocess
    import PIL.Image
    from skimage.measure import label, regionprops

    encoder_model = 'resnet18'
    n_class = 1
    weight_path = 'weights/best_weight.pth'
    warnings.filterwarnings('ignore')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if allGpus==True:
        model = smp.Unet(encoder_name=encoder_model, decoder_use_batchnorm=True, in_channels=3, classes=n_class)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model = smp.Unet(encoder_name=encoder_model, decoder_use_batchnorm=True,
                         in_channels=3, classes=n_class).to(device)
        model.load_state_dict(torch.load(weight_path))



    ###DataLoader
    val_dl = dataloader.DataLoader(data=extraction_coords,svs_path=path_svs,ImgNormalize=ImgNormalize, CaseName=CaseName,batch_size=batch_size, patch_size=patchsize,
                                   num_threads_in_multithreaded=1, seed_for_shuffle=5243,
                                   return_incomplete=True, shuffle=False, infinite=False)

    model.eval()
    swap = (lambda x: np.einsum('bchw->bhwc', x))
    repeat_channel = (lambda x: np.repeat(x, 3, axis=-1))
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            val_batch = next(val_dl)
            #imgs = val_batch
            imgs = val_batch['data']
            if len(imgs) == 0:
                continue
            imgs = utils.min_max_norm(imgs)
            imgs = torch.from_numpy(imgs).to(device)
            pred = model(imgs)
            pred_list = pred.cpu().detach().numpy()
            total_preds = repeat_channel(swap(np.where(np.vstack(tuple([pred_list])) > 0.5, 1.0, 0.0)))
            tileInfo = val_batch['tileInfo']
            tileCoords = val_batch['tileCoords']
#            imgs = imgs.cpu().detach().numpy()
            imgs = val_batch['data']
            PathGlandsMetaplasia = 'Dataset_GlandDetection/MetaplasiaGland/' + CaseName + '/'
            os.makedirs(PathGlandsMetaplasia, exist_ok=True)

            PathGlandsControl = 'Dataset_GlandDetection/ControlGland/' + CaseName + '/'
            os.makedirs(PathGlandsControl, exist_ok=True)

            if HighResol == False:
                if not os.path.exists(outputIm):
                    data = np.zeros((patchsize[0], patchsize[1], 3), dtype=np.uint8)
                    img = Image.fromarray(data, 'RGB')

                    # Guarda la imagen en un archivo
                    img.save('img_0_0.bmp')
                    subprocess.call(
                        ['/home/ricardo/kakadu/bin/kdu_compress', '-i', 'img_0_0.bmp', '-o', outputIm,
                         'Creversible=yes', 'Clevels=4', 'Stiles={' + str(patchsize[0]) + ',' + str(patchsize[1]) + '}',
                         'Clayers=10',
                         'Cprecincts={512,512},{256,256},{128,128},{64,64},{32,32}', 'Corder=LRCP',
                         'ORGgen_plt=yes', 'ORGtparts=R', 'Cuse_sop=yes', 'Cuse_precincts=yes', '-frag',
                         str(0) + ',' + str(0) + ',' + str(1) + ',' + str(1),
                         'Sdims={' + str(svs_size[1]) + ',' + str(svs_size[0]) + '}', 'Scomponents=3'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                for total_pred,tInfo,img,(x,y) in zip(total_preds,tileInfo,imgs,tileCoords):

                    #res = PIL.Image.fromarray(total_pred.astype(np.uint8)*255).resize(patchsize, Image.NEAREST)
                    res = PIL.Image.fromarray((total_pred * 255).astype(np.uint8)).resize(patchsize, Image.BILINEAR)
                    imgRgb = np.transpose(img, (1, 2, 0)).astype(int)
                    #pred_boundary_img = utils.overlay_boundary(imgRgb, total_pred)
                    #imgRgb = PIL.Image.fromarray((pred_boundary_img*255).astype(np.uint8)).resize(patchsize, Image.BILINEAR)
                    imgRgb = PIL.Image.fromarray((imgRgb).astype(np.uint8)).resize(patchsize,
                                                                                                   Image.BILINEAR)
                    labels, canvas = process_glands(MetaplasiaPredsCase, res, x, y)

                    MetaplasiaMaskName = CaseName + '_X_' + str(x) + '_Y_' + str(y) + '_' + str(tInfo[0]) + '_' + str(
                        tInfo[1]) + '.mat'
                    ControlMaskName = CaseName + '_X_' + str(x) + '_Y_' + str(y) + '_' + str(tInfo[0]) + '_' + str(
                        tInfo[1]) + '.mat'

                    labels = labels + 1  # se pone el fondo en 1 en los elementos
                    RemoveControl = canvas == 1  # Determinar los pixels control
                    IdxMetaplasia = labels.copy()  # duplicar labels
                    IdxMetaplasia[RemoveControl] = 0
                    #save IdxMetaplasia
                    #os.path.join(PathGlandsMetaplasia, MeplasiaMaskName)
                    #os.path.join(PathGlandsControl, ControlMaskName)
                    scipy.io.savemat(os.path.join(PathGlandsMetaplasia, MetaplasiaMaskName) , {"MaskL": IdxMetaplasia.astype(np.uint8)})

                    RemoveMetaplasia = canvas == 2  # Determinar los pixels control
                    IdxControl = labels.copy()  # duplicar labels
                    IdxControl[RemoveMetaplasia] = 0
                    scipy.io.savemat(os.path.join(PathGlandsControl, ControlMaskName), {"MaskL": IdxControl.astype(np.uint8)})

                    canvasRgb = utils.overlay_boundary(np.array(imgRgb), np.array(canvas == 1) * 1)
                    canvasRgb = utils.overlay_boundary(canvasRgb, np.array(canvas == 2) * 1 + 1, color=(1.0, 0, 0))

                    imgRgb = PIL.Image.fromarray((canvasRgb*255).astype(np.uint8)).resize(patchsize,
                                                                                                    Image.BILINEAR)
                    imgRgb.save('img_' + str(0) + '_' + str(0) + '.bmp')
                    ## kakadu
                    subprocess.call(
                        ['/home/ricardo/kakadu/bin/kdu_compress', '-i', 'img_' + str(0) + '_' + str(0) + '.bmp', '-o', outputIm,
                         'Creversible=yes', 'Clevels=4', 'Stiles={' + str(patchsize[0]) + ',' + str(patchsize[1]) + '}', 'Clayers=10',
                         'Cprecincts={512,512},{256,256},{128,128},{64,64},{32,32}', 'Corder=LRCP',
                         'ORGgen_plt=yes', 'ORGtparts=R', 'Cuse_sop=yes', 'Cuse_precincts=yes', '-frag',
                         str(tInfo[1]) + ',' + str(tInfo[0]) + ',' + str(1) + ',' + str(1), 'Sdims={' + str(svs_size[1]) + ',' + str(svs_size[0]) + '}', 'Scomponents=3'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                for total_pred, tInfo, img, (x, y) in zip(total_preds, tileInfo, imgs, tileCoords):
                    # res = PIL.Image.fromarray(total_pred.astype(np.uint8)*255).resize(patchsize, Image.NEAREST)
                    res = PIL.Image.fromarray((total_pred * 255).astype(np.uint8)).resize(patchsize, Image.BILINEAR)
                    imgRgb = slide.read_region((x, y), 0, patchsize)
                    # imgRgb = np.transpose(img, (1, 2, 0)).astype(int)
                   # pred_boundary_img = utils.overlay_boundary(np.array(imgRgb)[:, :, 0:3], np.array(res) )
                   # imgRgb = PIL.Image.fromarray((pred_boundary_img * 255).astype(np.uint8))#.resize(patchsize,Image.BILINEAR)

                    imgRgb.save('img_' + str(0) + '_' + str(0) + '.bmp')
                    ### kakadu

                    subprocess.call(
                        ['/home/ricardo/kakadu/bin/kdu_compress', '-i', 'img_' + str(0) + '_' + str(0) + '.bmp', '-o',
                         outputIm,
                         'Creversible=yes', 'Clevels=4', 'Stiles={' + str(patchsize[0]) + ',' + str(patchsize[1]) + '}',
                         'Clayers=10',
                         'Cprecincts={512,512},{256,256},{128,128},{64,64},{32,32}', 'Corder=LRCP',
                         'ORGgen_plt=yes', 'ORGtparts=R', 'Cuse_sop=yes', 'Cuse_precincts=yes', '-frag',
                         str(tInfo[1]) + ',' + str(tInfo[0]) + ',' + str(1) + ',' + str(1),
                         'Sdims={' + str(svs_size[1]) + ',' + str(svs_size[0]) + '}', 'Scomponents=3'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            #### Save Glands and Determine if is metaplasia or not
            #for glands
            # save Glands


# path_svs = 'U001.svs'
# patchsize=(512*4,512*4)
# batch_size=50
# HighResol = False
# ImgNormalize= False
# outputIm='prueba2048NormalizedMetaplasiasolving.jpc'
# MetaplasiaPredsCase = MetaplasiaPreds.loc['U001']
# generate_WSI(patchsize, batch_size, path_svs, outputIm=outputIm, HighResol=HighResol, ImgNormalize=ImgNormalize,
#
#
#                      MetaplasiaPredsCase=MetaplasiaPredsCase,CaseName='U001')
import os


def process_all_images(path_folder, patchsize, batch_size, HighResol, ImgNormalize, MetaplasiaPreds):
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
        MetaplasiaPredsCase = MetaplasiaPreds.loc[case_name]

        ####################################################################
        ###### Sacar parches para deteccion de Metaplasia Vs. Control ######
        ####################################################################






        #########################################################
        ####### sacar Parches para deteccion de glandulas #######
        #########################################################




        ##########################################################



        generate_WSI(patchsize, batch_size, path_svs, outputIm=outputIm, HighResol=HighResol, ImgNormalize=ImgNormalize,
                     MetaplasiaPredsCase=MetaplasiaPredsCase,CaseName=case_name)


# Uso de la función
path_folder = '/home/ricardo/Descargas/DatosUrk/'
patchsize = (512 * 4, 512 * 4)
batch_size = 50
HighResol = False
ImgNormalize = True

process_all_images(path_folder, patchsize, batch_size, HighResol, ImgNormalize, MetaplasiaPreds)
