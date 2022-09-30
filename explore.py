# https://www.kaggle.com/getting-started/141256

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# %matplotlib inline
# import matplotlib.patches as patches
# import seaborn as sns
# sns.set(style='darkgrid', font_scale=1.6)
# import cv2
import os
# from os import listdir
# import re
# import gc
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
# from tqdm import tqdm
# from pprint import pprint
# from time import time
# import itertools
# from skimage import measure 
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
from glob import glob
import plotly.express as px

from IPython.display import display_html
# plt.rcParams.update({'font.size': 16})

# import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=FutureWarning)


# Custom colors
class clr:
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'



# Base path
base_path = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"

# Load metadata
# train_df = pd.read_csv(f"{base_path}/train.csv")
# train_bbox = pd.read_csv(f"{base_path}/train_bounding_boxes.csv")
# test_df = pd.read_csv(f"{base_path}/test.csv")
# ss = pd.read_csv(f"{base_path}/sample_submission.csv")

# ===========================================================================================================================================
# ==================== READ DATA  ===========================================================================================================
# ===========================================================================================================================================

def read_data():
    '''Reads in all .csv files.'''
    
    # Load metadata
    train_df = pd.read_csv(f"{base_path}/train.csv")
    train_bbox = pd.read_csv(f"{base_path}/train_bounding_boxes.csv")
    test_df = pd.read_csv(f"{base_path}/test.csv")
    ss = pd.read_csv(f"{base_path}/sample_submission.csv")
    
    return train_df, train_bbox, test_df, ss

def read_folders():
    '''Reads folders and files. TO DO'''
    
    # Load metadata
    
    return 1


def get_csv_info(csv, name="Default"):
    '''Prints main information for the speciffied .csv file.'''
    
    print(clr.S+f"=== {name} ==="+clr.E)
    print(clr.S+f"Shape:"+clr.E, csv.shape)
    print(clr.S+f"Missing Values:"+clr.E, csv.isna().sum().sum(), "total missing datapoints.")
    print(clr.S+"Columns:"+clr.E, list(csv.columns), "\n")
    
    display_html(csv.head())
    print("\n")


def display_metadata():

    # Print parameters
    print('base_path = {base_path}')
    print('')

    # Read in the data
    train, train_bbox, test, ss = read_data()

    # Print useful information on it
    for csv, name in zip([train, train_bbox, test, ss],
                        ["Train", "Train Bbox", "Test", "Sample Submission"]):
        get_csv_info(csv, name)

# ===========================================================================================================================================
# ==================== MERGE DATA  ==========================================================================================================
# ===========================================================================================================================================

def merge_data():
    '''Merge all data information. TO DO'''
    
    # Load metadata
    
    
    return 1


# ===========================================================================================================================================
# ==================== EXPLORE ==============================================================================================================
# ===========================================================================================================================================

def get_patient_info(patient_id):

    # ===== Retrieve dcms ===========================
    dcms_path = glob(f"{base_path}/train_images/{patient_id}/*.dcm")
    slice_num = [int(p.split('/')[-1][:-4]) for p in dcms_path]
    dcms_path = [element for _,element in sorted(zip(slice_num, dcms_path))] # sort

    dcms=[]
    for dcm_path in dcms_path:
        file = pydicom.dcmread(dcm_path)
        img = apply_voi_lut(file.pixel_array, file)
        dcms.append(img)

    dcms=np.array(dcms)
    print('DICOM images shape:',dcms.shape)
    
    # ===== retrieve patient segmentations ========== 
    seg_path = f"{base_path}/segmentations/{patient_id}.nii"
    if os.path.exists(seg_path):
        nii_example = nib.load(seg_path)
        segs = nii_example.get_fdata()
        # Align orientation with images
        segs = segs[:, ::-1, ::-1].transpose(2, 1, 0)
        print('Segmentation shape:',segs.shape)
        pos=[]
        for i in range(segs.shape[0]):
            arr=list(np.unique(segs[i]).astype('int'))
            arr.pop(0)
            zones = len(arr)
            if zones!=0:
                if np.min(arr)<=7:
                    pos.append([i,arr])
        min_max_seg_slices=(pos[1][0],pos[-1][0])
    else:
        print('No segmentation')
        segs=np.ones(dcms.shape)
        min_max_seg_slices=(0,0)
    
        
    # ===== Retrieve bouding boxes ==================
    # bboxs= np.zeros(dcms.shape)
    # n=dcms.shape[0] # number of slices
    # bbox_df = train_bbox[train_bbox['StudyInstanceUID']==patient_id]
    
    bbox_coord=[]
#     if len(bbox_df)!=0:
#         for i in bbox_df.index:
#             slice_n = bbox_df.loc[i,'slice_number']
#             x,y,w,h = float(info.x), float(info.y), float(info.width), float(info.height)
#             xbox=[x,x+w,x+w,x,x]
#             ybox=[y,y,y+h,y+h,y]
#             bbox_coord.append([slice_n,x,y])
# #             bboxs=go.Scatter(x=xbox, y=ybox, fill="toself")
#         bbox_coord=np.array(bbox_coord) 
#         print('Bounding boxes:',bbox_coord.shape)
#     else:
#         print('No bouding boxes.')

    return dcms, segs, min_max_seg_slices, bbox_coord


def show_patient(patient_id,x=0.1):
    
    dcms, segs, min_max_seg_slices, bbox_coord = get_patient_info(patient_id)
    print(f"min and max of slices segmented: ",min_max_seg_slices)
    dcms_segmented = segs * dcms
    imgs = np.concatenate([dcms, dcms_segmented],axis =2)
    
    start = min_max_seg_slices[0]
    end = min_max_seg_slices[1]

    
    fig = px.imshow(
    #                 segs[int((end-start)/2-x*(end+start)/2):int((end-start)/2+x*(end+start)/2)]*255,
                    imgs[int((end-start)/2-x*(end+start)/2):int((end-start)/2+x*(end+start)/2)],
    #                 imgs[start:end],
    #                 color_continuous_scale='gray',
                    cmap='inferno',
                    animation_frame=0,
                    binary_string=True,
                    labels=dict(animation_frame="slice")
                   )

    # fig = go.Figure(go.Scatter(x=xbox, y=ybox, fill="toself"))

    fig.update_traces(hovertemplate="x: %{x} <br> y: %{y} <br> z: %{z}")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()