# https://www.kaggle.com/getting-started/141256

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib.patches as patches
# import seaborn as sns
# sns.set(style='darkgrid', font_scale=1.6)
# import cv2
import os
# from os import listdir
# import re
# import gc
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm
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

def read_csv():
    '''Reads all .csv files.'''
    
    # Load metadata
    train_df = pd.read_csv(f"{base_path}/train.csv")
    train_bbox = pd.read_csv(f"{base_path}/train_bounding_boxes.csv")
    test_df = pd.read_csv(f"{base_path}/test.csv")
    ss = pd.read_csv(f"{base_path}/sample_submission.csv")
    
    return train_df, train_bbox, test_df, ss

def read_dcms(type='train'):
    '''Reads dcms filenames and stores it under a dataframe.
    type = 'train' or 'test'
       --> scan either base_path/train_images or base_path/tesf_images
    '''

    dcms_path = glob(f"{base_path}/{type}_images/*/*.dcm")
    dcms_df = pd.DataFrame({'path': dcms_path})
    dcms_df['StudyInstanceUID'] = dcms_df['path'].apply(lambda x:x.split('/')[-2])
    dcms_df['slice_number'] = dcms_df['path'].apply(lambda x: int(x.split('/')[-1][:-4]))
    dcms_df = dcms_df[['StudyInstanceUID','slice_number']].sort_values(by=['StudyInstanceUID','slice_number'])
    # print('dcms_df shape:', dcms_df.shape)
    # display(dcms_df.head(3))
    dcms_summary = dcms_df.groupby('StudyInstanceUID').agg({'slice_number':'count'}).reset_index().rename(columns={'slice_number':'nb_of_slices'})
    # print('train_imgs_summary shape:', dcms_summary.shape)
    # display(dcms_summary.head(3))
    
    return dcms_df, dcms_summary

def read_seg(option='detailed'):
    '''Reads segmentation files and stores it under a dataframe.
    '''

    if option=='detailed':

        patient_id=[]
        shapes=[]
        nb_slices=[]
        # min_max_slices=[]
        # min_slice=[]
        # max_slice=[]
        for dirname, _, files in tqdm(os.walk(f"{base_path}/segmentations")):
            for filename in files:
                patient_id.append(filename.replace('.nii',''))
                nii_example = nib.load(os.path.join(dirname, filename))
                # Convert to numpy array
                seg = nii_example.get_fdata()
                # Align orientation with images
                shapes.append(seg.shape)
                nb_slices.append(seg.shape[2])
                
                # Align orientation with images
        #         seg = seg[:, ::-1, ::-1].transpose(2, 1, 0)
                
        #         pos=[]
        #         for i in range(seg.shape[0]):
        #             arr=list(np.unique(seg[i]).astype('int'))
        #             arr.pop(0)
        #             zones = len(arr)
        #             if zones!=0:
        #                 if np.min(arr)<=7:
        #                     pos.append([i,arr])
                            
        #         min_max_slices.append([pos[1],pos[-1]])
        #         min_slice.append(pos[1][0])
        #         max_slice.append(pos[-1][0])
    
        seg_df = pd.DataFrame({'StudyInstanceUID':patient_id,
                                    'mask_shape':shapes,
                                    'mask_nb_slices':nb_slices,
                                    'isSegmented':1,
        #                                'mask_min_max_slices':min_max_slices,
        #                                'mask_min_slice':min_slice,
        #                                'mask_max_slice':max_slice
                                    })

    else:

        # Store segmentation paths in a dataframe
        seg_paths = glob(f"{base_path}/segmentations/*")
        seg_df = pd.DataFrame({'path': seg_paths})
        seg_df['StudyInstanceUID'] = seg_df['path'].apply(lambda x:x.split('/')[-1][:-4])
        seg_df['isSegmented'] = 1
        seg_df = seg_df[['StudyInstanceUID','isSegmented']]
    
    return seg_df


def get_df_info(df, name="Default"):
    '''Prints main information for the speciffied .csv file.'''
    
    print(clr.S+f"=== {name} ==="+clr.E)
    print(clr.S+f"Shape:"+clr.E, df.shape)
    print(clr.S+f"Missing Values:"+clr.E, df.isna().sum().sum(), "total missing datapoints.")
    print(clr.S+"Columns:"+clr.E, list(df.columns), "\n")
    
    display_html(df.head())
    print("\n")


def display_metadata():

    # Print parameters
    print('base_path = {base_path}')
    print('')

    # Read in the data
    train, train_bbox, test, ss = read_csv()

    # Print useful information on it
    for df, name in zip([train, train_bbox, test, ss],
                        ["Train", "Train Bbox", "Test", "Sample Submission"]):
        get_df_info(df, name)

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


def show_patient(patient_id,slice_range=0.1):

    '''
    Plot patient DCMS and Segmentation images
    slice_range = % within the segmentation images
    '''
    
    dcms, segs, min_max_seg_slices, bbox_coord = get_patient_info(patient_id)

    if min_max_seg_slices==(0,0):
        print("No segmentation")
        start, end = 0, dcms.shape[0]
    else:
        print("min and max of slices segmented: ",min_max_seg_slices)
        start, end = min_max_seg_slices[0], min_max_seg_slices[1]
    
    slider_slices = int((end-start)/2-slice_range/2*(end+start)/2),int((end-start)/2+slice_range/2*(end+start)/2)
    print("slider range (min slice num, max slice num): ",slider_slices)


    dcms_segmented = segs * dcms
    imgs = np.concatenate([dcms, dcms_segmented],axis =2)
    
    fig = px.imshow(
                    # segs[slider_slices[0]:slider_slices[1]],
                    imgs[slider_slices[0]:slider_slices[1]],
    #                 imgs[start:end],
    #                 color_continuous_scale='gray',
                    # cmap='inferno',
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



def plot_fractures(patient_id, nb_columns = 2):

    train_df , train_bbox, _, _  = read_data()
    
    bbox= train_bbox[train_bbox.StudyInstanceUID==patient_id]
    n = len(bbox)

    display(train_df[train_df.StudyInstanceUID==patient_id])
    
    fig, axes = plt.subplots(nrows= (n // nb_columns), ncols= nb_columns, figsize=(nb_columns*12,((n // nb_columns)+1)*12))
    
    for k,i in enumerate(bbox.index):
        slice_num=bbox.loc[i,'slice_number']
        file = pydicom.dcmread(f"{base_path}/train_images/{patient_id}/{slice_num}.dcm")
        img = apply_voi_lut(file.pixel_array, file)
        info = train_bbox[(train_bbox['StudyInstanceUID']==patient_id)&(train_bbox['slice_number']==slice_num)]
        rect = patches.Rectangle((float(info.x), float(info.y)), float(info.width), float(info.height), linewidth=3, edgecolor='r', facecolor='none')
        
        ax_id1 = k // nb_columns
        ax_id2 = k % nb_columns
        axes[ax_id1,ax_id2].imshow(img, cmap="bone")
        axes[ax_id1,ax_id2].add_patch(rect)
#         axes[ax_id1,ax_id2].set_title(f"ID:{patient_id}, Slice: {slice_num}", fontsize=20, weight='bold',y=1.02)
        axes[ax_id1,ax_id2].set_title(f"Slice:{slice_num}", fontsize=20, weight='bold',y=1.02)
        axes[ax_id1,ax_id2].axis('off')