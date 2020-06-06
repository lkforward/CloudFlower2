import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import albumentations as albu


## HELPER FUNCTIONS

# Function 1. To get the image and its masks from the image name.
def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    if mask_rle == '' or (type(mask_rle) is not str):
        return None
    # Creat the mask as 1D first; we will reshape it at the end.
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    rle_code = mask_rle.split()

    # for each segment of the mask:
    i = 0
    while i + 1 < len(rle_code):
        ibgn = int(rle_code[i])
        seg_len = int(rle_code[i + 1])
        mask[ibgn: ibgn + seg_len] = 1
        i += 2

    # Note: the order of rle_code is row first (from top to bottom), and then column.
    return mask.reshape(shape, order='F')


def plot_image_with_masks(img, mask_dict, image_name=''):
    """
    img: numpy array. 
      The image object read by cv2.imread().
    mask_dict: dictionary. 
      keys: the mask label.
      values: the correspondign mask as numpy array. 
    """
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))

    for ax, l in zip(axs.flatten(), list(mask_dict.keys())):
        ax.imshow(img)

        if mask_dict[l] is not None:
            ax.imshow(mask_dict[l], alpha=0.3, cmap='gray')

        if len(image_name) > 0:
            ax.set_title(image_name + ':' + l)
        else:
            ax.set_title(l)
        ax.grid(True)

def viz_image_mask_arrays(img_array, mask_arrays):
    """
    Visualize an image and the four masks in numpy array format. 
    
    :param img_array: A numpy array with the shape of (M, N, 3)
    :param mask_arrays: A numpy array with the shape of (M, N, 4). 
    :return: 
    """
    assert(img_array.ndim == 3 and img_array.shape[-1]==3), "The image array should have a shape of (M, N, 3)!"
    assert(mask_arrays.ndim == 3 and mask_arrays.shape[-1]==4), "The mask array should have a shape of (M, N, 4)!"

    fig, axs = plt.subplots(2, 2, figsize=(18, 10))

    for i in range(4):
        ax = axs.flatten()[i]
        ax.imshow(img_array)
        ax.imshow(mask_arrays[:,:,i], alpha=0.3, cmap='gray')
        ax.grid(True)



def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.augmentations.transforms.Lambda(image=preprocessing_fn),
        albu.augmentations.transforms.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_training_augmentation():
    """
    Define the preprocessing for the training data. 
    """
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        albu.Resize(320, 640)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(320, 640)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')


def split_image_dataset(data_df, train_ratio, max_n_images=None):
    """
    data_df: dataframe. 
      Read from the file train.csv.
    train_ratio: float.
      A float between 0 and 1. The ratio of data used for the training. The rest 
      will be used for validation. 
    max_n_images: int. 
      The maximum number used for the training/validation. If this argument is not 
      specified or the number of images in data_df is amller than max_n_images, 
      the whole dataset is used. 
  
    return: 
    train_df / valid_df: dataframe. 
      Containing the images for training / validation purpose. 
    """
    # 4:15

    assert ('image_name' in data_df.columns), "The input dataframe should have a col for image_name!"
    img_names = list(data_df['image_name'].unique())

    if (max_n_images is not None) and (len(img_names) > max_n_images):
        img_names = img_names[:max_n_images]

    from sklearn.model_selection import train_test_split
    train_names, valid_names = train_test_split(img_names,
                                                train_size=train_ratio,
                                                random_state=42)

    train_df = data_df.loc[data_df['image_name'].isin(train_names)]
    valid_df = data_df.loc[data_df['image_name'].isin(valid_names)]

    return train_df, valid_df


def read_train_df(csv_fname):
    get_name = lambda x: x.split('_')[0]
    get_label = lambda x: x.split('_')[1]

    train = pd.read_csv(csv_fname)
    train['image_name'] = train.Image_Label.apply(get_name)
    train['label'] = train.Image_Label.apply(get_label)

    return train