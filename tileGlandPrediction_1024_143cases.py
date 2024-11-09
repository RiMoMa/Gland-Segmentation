import numpy as np
from PIL import Image
import openslide
import os
import glob

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
        match = re.search(r'_X_(\d+)', image_name)
        # Si se encuentra una coincidencia, convertir el resultado en un entero. Si no, devolver None
        x_coord = int(match.group(1)) if match else None

        # Buscar las coordenadas y en el nombre de la imagen y guardar el resultado
        match = re.search(r'_Y_(\d+)', image_name)
        # Si se encuentra una coincidencia, convertir el resultado en un entero. Si no, devolver None
        y_coord = int(match.group(1)) if match else None

        # Devolver las coordenadas x e y como una tupla
        return x_coord, y_coord

    # Añadir una nueva columna al DataFrame con los nombres de las imágenes extraídos de la columna 'name'
    MetaplasiaPreds['image_name'] = MetaplasiaPreds['name'].str.split('/').str[-1]

    # Añadir una nueva columna al DataFrame con los nombres de los casos extraídos de la columna 'image_name'
    casename = re.search(r'(\d{3}_\d{3}_\d{4}_[a-z]{2})', file_path).group(1)

    MetaplasiaPreds['case_name'] = casename

    # Aplicar la función extract_coordinates a la columna 'image_name' y dividir el resultado en dos nuevas columnas
    MetaplasiaPreds['x_coord'], MetaplasiaPreds['y_coord'] = zip(
        *MetaplasiaPreds['image_name'].apply(extract_coordinates))

    # Convertir las columnas 'x_coord' y 'y_coord' en enteros
    print(MetaplasiaPreds['x_coord'].head())
    MetaplasiaPreds['x_coord'] = MetaplasiaPreds['x_coord'].astype(int)
    MetaplasiaPreds['y_coord'] = MetaplasiaPreds['y_coord'].astype(int)

    # Cambiar el índice del DataFrame a la columna 'case_name'
    MetaplasiaPreds.set_index('case_name', inplace=True)

    # Devolver el DataFrame
    return MetaplasiaPreds

# Uso de la función
#file_path = '/home/ricardo/Documentos/IsbiAutomatico/Dataset/convnext.data.csv'
#MetaplasiaPreds = preprocess_data(file_path)

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
#######

def gland_predictions(files_path=None,patchsize=(512,512), batch_size=100,probValueF=0.5,path_svs = None, allGpus=False,ImgNormalize=False, MetaplasiaPredsCase=None,CaseName=None,OutputOverlay=None, OutputGlandPath=None):
    # open svs

    #hacer el forde las predicciones del modelo generar las predicciones y luego con kakadu generar el WSI
    num_batches = math.ceil(len(files_path) / batch_size)
    import scipy.io

    ##### Cargar MODELO UNET #####
    import warnings
    import dataloaderTileGlands as dataloader
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
    device = torch.device("cuda:0")

    if allGpus==True:
        model = smp.Unet(encoder_name=encoder_model, decoder_use_batchnorm=True, in_channels=3, classes=n_class)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model = smp.Unet(encoder_name=encoder_model, decoder_use_batchnorm=True,
                         in_channels=3, classes=n_class).to(device)

        # Modificar la U-Net para devolver las características del bottleneck
        class UNetWithBottleneck(smp.Unet):
            def forward(self, x):
                # Encoder: capturar el resultado de cada etapa
                encoder_features = self.encoder(x)
                bottleneck = encoder_features[-1]  # Cuello de botella
                decoder_output = self.decoder(*encoder_features)
                masks = self.segmentation_head(decoder_output)
                return masks, bottleneck

        # Crear el modelo modificado
        model = UNetWithBottleneck(encoder_name=encoder_model, decoder_use_batchnorm=True,
                                   in_channels=3, classes=n_class).to(device)
        model.load_state_dict(torch.load(weight_path))



    ###DataLoader
    val_dl = dataloader.DataLoader(data=files_path,svs_path=path_svs,ImgNormalize=ImgNormalize, CaseName=CaseName,batch_size=batch_size, patch_size=patchsize,
                                   num_threads_in_multithreaded=1, seed_for_shuffle=5243,
                                   return_incomplete=True, shuffle=False, infinite=False,
                                   inputDataset=None)

    model.eval()
    swap = (lambda x: np.einsum('bchw->bhwc', x))
    repeat_channel = (lambda x: np.repeat(x, 3, axis=-1))
    bottleneck_list = []  # Para guardar los valores del bottleneck

    with (torch.no_grad()):
        for i in tqdm(range(num_batches)):
            val_batch = next(val_dl)
            #imgs = val_batch
            imgs = val_batch['data']
            if len(imgs) == 0:
                continue
            imgs = utils.min_max_norm(imgs)
            imgs = torch.from_numpy(imgs).to(device)
            #pred = model(imgs)
            pred, bottleneck = model(imgs)
            bottleneck = bottleneck.cpu().detach().numpy()
            pred_list = pred.cpu().detach().numpy()
            total_preds = repeat_channel(swap(np.where(np.vstack(tuple([pred_list])) > probValueF , 1.0, 0.0)))

            imgs = val_batch['data']
            tileInfo = val_batch['tileInfo']

            PathGlandsMetaplasia = OutputGlandPath+'/MetaplasiaGland/' + CaseName + '/'
            os.makedirs(PathGlandsMetaplasia, exist_ok=True)

            PathGlandsControl = OutputGlandPath+'/ControlGland/' + CaseName + '/'
            os.makedirs(PathGlandsControl, exist_ok=True)

            for total_pred,img,tileName,bottleneck_img in zip(total_preds,imgs,tileInfo,bottleneck):

                #res = PIL.Image.fromarray(total_pred.astype(np.uint8)*255).resize(patchsize, Image.NEAREST)
                res = PIL.Image.fromarray((total_pred * 255).astype(np.uint8)).resize(patchsize, Image.BILINEAR)
                imgRgb = np.transpose(img, (1, 2, 0)).astype(int)
                #pred_boundary_img = utils.overlay_boundary(imgRgb, total_pred)
                #imgRgb = PIL.Image.fromarray((pred_boundary_img*255).astype(np.uint8)).resize(patchsize, Image.BILINEAR)
                imgRgb = PIL.Image.fromarray((imgRgb).astype(np.uint8)).resize(patchsize,Image.BILINEAR)
               # x = int(re.search(r"_X_(\d+)_Y_", tileName).group(1))
              #  y = int(re.search(r"_Y_(\d+)_", tileName).group(1))
             #   tInfo = re.search(r"_([^_]*)_([^_]*)\.png$", tileName).groups()
            #    tInfo = [int(i) for i in tInfo]
                #labels, canvas = process_glands(MetaplasiaPredsCase, res, x, y)
              #  MetaplasiaMaskName = CaseName + '_X_' + str(x) + '_Y_' + str(y) + '_' + str(tInfo[0]) + '_' + str(
                #    tInfo[1]) + '.mat'
               # ControlMaskName = CaseName + '_X_' + str(x) + '_Y_' + str(y) + '_' + str(tInfo[0]) + '_' + str(
                 #   tInfo[1]) + '.mat'

                #labels = labels + 1  # se pone el fondo en 1 en los elementos
                #RemoveControl = canvas == 1  # Determinar los pixels control
                #IdxMetaplasia = labels.copy()  # duplicar labels
                #IdxMetaplasia[RemoveControl] = 0
                # save IdxMetaplasia

                #scipy.io.savemat(os.path.join(PathGlandsMetaplasia, MetaplasiaMaskName),
                 #                {"MaskL": IdxMetaplasia.astype(np.uint8)})

               # RemoveMetaplasia = canvas == 2  # Determinar los pixels control
                #IdxControl = labels.copy()  # duplicar labels
                #IdxControl[RemoveMetaplasia] = 0
                #scipy.io.savemat(os.path.join(PathGlandsControl, ControlMaskName), {"MaskL": IdxControl.astype(np.uint8)})

                #canvasRgb = utils.overlay_boundary(np.array(imgRgb), np.array(canvas == 1) * 1)
                #canvasRgb = utils.overlay_boundary(canvasRgb, np.array(canvas == 2) * 1 + 1, color=(1.0, 0, 0))
                canvasRgb = utils.overlay_boundary(np.array(imgRgb),np.array(res))
                imgRgb = PIL.Image.fromarray((canvasRgb*255).astype(np.uint8)).resize(patchsize,
                                                                                                Image.BILINEAR)
                filename = os.path.basename(tileName)
                outputImagePath = OutputOverlay + CaseName+'/'
                os.makedirs(outputImagePath,exist_ok=True)
                FolderNameAll =tileName.split('/')
                FolderName =FolderNameAll[6]# adjust the index as needed based on your path
                PathBotleneck = os.path.join(OutputGlandPath,'npy',FolderName)
                os.makedirs(PathBotleneck,exist_ok=True)
                PathBotleneck_img = os.path.join(PathBotleneck,FolderNameAll[-1][:-4]+'.npy')
                np.save(PathBotleneck_img, bottleneck_img.ravel())
                imgRgb.save(outputImagePath+filename)



import os

import subprocess
def process_all_images(path_folder, patchsize, batch_size, HighResol, ImgNormalize,
                       OutputOverlayGland,OutputGlandPath,probValue=0.5):
    # Listar todos los archivos en la carpeta
    all_files = os.listdir(path_folder)
    # Procesar cada archivo
    for svs_file in all_files:#svs_files:
        #svs_file = '001_194_2019_he.svs' #todo: borrar esto

        path_svs = os.path.join(path_folder, svs_file)
        #outputIm = svs_file.replace('.svs', '.jpc')

        # La última parte toma en cuenta solo el nombre del archivo
        case_name = svs_file
        ### Folder With metaplasia prediction control
     #   file_path = '/media/ricardo/84C86C94C86C8670/PredictionsGastrico/'+case_name+'.test.csv'#001_010_2019_he_001.test.csv'
      #  MetaplasiaPreds = preprocess_data(file_path)
       # MetaplasiaPredsCase = MetaplasiaPreds


        ####################################################################
        ###### Sacar parches para deteccion de Metaplasia Vs. Control ######
        ####################################################################
        input_path = path_svs+ '/*.png'
        input_files = glob.glob(input_path)

        # Obtén el nombre base de los archivos en ambos directorios
      #  input_files = [os.path.basename(file) for file in input_files]


        gland_predictions(files_path=input_files,patchsize=patchsize,batch_size= batch_size,path_svs= path_svs,
                          probValueF=probValue,
                 MetaplasiaPredsCase=None,
                 CaseName=case_name,
                 OutputOverlay = OutputOverlayGland,
                          OutputGlandPath=OutputGlandPath)




####################CALL 1 Inputs #################


# Uso de la función
ExpProbabilities = [2.5,0.5,-2]
for ExpProbability in ExpProbabilities:
    print(ExpProbability)
    probValue = ExpProbability
    path_folder = '/media/ricardo/Datos/Data/Processed_PatchesNormalized/'
    OutputOverlayGland = '/media/ricardo/Datos/Data/Processed_Patches_GlandPredictionOverlayNormalized'+str(ExpProbability)+'/'
    OutputGlandPath = '/media/ricardo/Datos/Data/Processed_Patches_GlandDatasetNormalizedNormalizad'+str(ExpProbability)+'/'
    patchsize = (512 , 512 )
    batch_size = 20
    HighResol = False
    ImgNormalize = True

    process_all_images(path_folder, patchsize, batch_size,
                       HighResol, ImgNormalize,
                       OutputOverlayGland=OutputOverlayGland,OutputGlandPath =OutputGlandPath,probValue=ExpProbability)
