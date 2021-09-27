import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

import geopandas
import rasterio
from rasterio.mask import mask


RAW_DATA_PATH = 'data'
LABELS_PATH = 'data/final_data.CSV'
PROCESSED_DATA_PATH = 'processed_data/'


def custom_reshape(img,window_size=512,stride=256):
    '''
    Pads the input image to accomodate running window of given size and stride.
    Padding is done by adding 0s to the left and bottom parts of the image.

    Parameters
    ----------
    img : numpy.ndarray
        2D array representing a channel of an RGB image.

    window_size : int, optional
        Size of the running window.

    stride : int, optional
        Number of pixels the given window shifts over the input image at a time.

    Returns
    -------
    2D numpy array of the padded image.
    '''
    h, w = img.shape

    h_diff = (h-window_size)%stride
    if h_diff!=0:
        pad_h = stride - h_diff
    else:
        pad_h = 0 # padding is not needed

    w_diff = (w-window_size)%stride
    if w_diff!=0:
        pad_w = stride - w_diff
    else:
        pad_w = 0 # padding is not needed

    img_new = img
    if pad_w!=0:
        img_new = np.concatenate((img_new, np.zeros((h,pad_w))),axis=1)
    if pad_h!=0:
        img_new = np.concatenate((img_new, np.zeros((pad_h,img_new.shape[1]))),axis=0)

    return img_new


def split_into_blocks(img,window_size=512,stride=256):
    '''
    Splits the input image into blocks of the given window size.

    Parameters
    ----------
    img : numpy.ndarray
        3D array representing an RGB image.

    window_size : int, optional
        Size of the running window.

    stride : int, optional
        Number of pixels the given window shifts over the input image at a time.

    Returns
    -------
    3D numpy array containing cropped image blocks. 
    '''
    r_channel = custom_reshape(img[:,:,0],window_size,stride)
    g_channel = custom_reshape(img[:,:,1],window_size,stride)
    b_channel = custom_reshape(img[:,:,2],window_size,stride)

    h, w = r_channel.shape

    blocks = []
    for i in range(0,h-window_size+1,stride):
        for j in range(0,w-window_size+1,stride):
            crop_r_channel = r_channel[i:i+window_size,j:j+window_size]
            crop_g_channel = g_channel[i:i+window_size,j:j+window_size]
            crop_b_channel = b_channel[i:i+window_size,j:j+window_size]

            I = np.dstack((crop_r_channel,crop_g_channel,crop_b_channel))

            no_0s = (I==0).sum()
            count = I.size

            if (no_0s/count)<0.5: # filtering out noisy blocks
                blocks.append(I)

    return np.array(blocks)


def write_blocks_to_disk(window_size=512,stride=256,p=0.2):
    '''
    Splits raw data into smaller blocks of images and saves as numpy arrays.
    Creates also two csv files containing the paths and labels of train and test images.

    Parameters
    ----------
    window_size : int, optional
        Size of the running window.

    stride : int, optional
        Number of pixels the given window shifts over the input image at a time.

    p : float, optional
        Proportion of images to keep as a test set. 
    '''
    labels = pd.read_csv(LABELS_PATH)

    Path(PROCESSED_DATA_PATH).mkdir(exist_ok=True)

    for folder in tqdm(sorted(glob.glob(os.path.join(RAW_DATA_PATH, '[!final_data]*')))):
        boundary = geopandas.read_file(os.path.join(folder, 'boundary.zip')) # boundary mask
        img_r = rasterio.open(os.path.join(folder, 'reflectance_red-red.tif')) # red channel
        img_g = rasterio.open(os.path.join(folder, 'reflectance_green-green.tif')) # green channel
        img_b = rasterio.open(os.path.join(folder, 'reflectance_blue-blue.tif')) # blue channel

        boundary = boundary['geometry'].to_crs(img_r.crs) # transform geometries to the image's coordinate reference system.

        img_r, _ = mask(img_r, boundary, crop=True)
        img_g, _ = mask(img_g, boundary, crop=True)
        img_b, _ = mask(img_b, boundary, crop=True)

        # normalizing channels
        img_r = np.clip(img_r[0]/(2**15),0,1)
        img_g = np.clip(img_g[0]/(2**15),0,1)
        img_b = np.clip(img_b[0]/(2**15),0,1)

        img_rgb = np.dstack((img_r,img_g,img_b))

        flight_code = folder.split('/')[1]
        status = labels[labels.flight_code==flight_code]['ActualStatus'].item()

        img_blocks = split_into_blocks(img_rgb,window_size=window_size,stride=stride)
        binom_mask = np.random.binomial(1,p,size=img_blocks.shape[0]) # mask for a train-test split

        df_train = pd.DataFrame(columns=['img','label'])
        df_test = pd.DataFrame(columns=['img','label'])

        # saving train data
        for i, sample in enumerate(img_blocks[np.where(binom_mask==0)[0]]):
            img_name = flight_code + '_' + str(i) + '_train.npy'
            np.save(os.path.join(PROCESSED_DATA_PATH, img_name), sample)
            df_train = df_train.append({'img':img_name,'label':status},ignore_index = True)

        # saving test data
        for i, sample in enumerate(img_blocks[np.where(binom_mask==1)[0]]):
            img_name = flight_code + '_' + str(i) + '_test.npy'
            np.save(os.path.join(PROCESSED_DATA_PATH, img_name), sample)
            df_test = df_test.append({'img':img_name,'label':status},ignore_index = True)


    df_train.to_csv(os.path.join(PROCESSED_DATA_PATH, 'train.csv'), encoding='utf-8', index=False)
    df_test.to_csv(os.path.join(PROCESSED_DATA_PATH, 'test.csv'), encoding='utf-8', index=False)

