import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import pandas as pd
from pytu.tensor_list import TensorList
import os
from random import sample 
from pathlib import Path
from sklearn.model_selection import train_test_split

seed = 1

def center_crop_img(img, cropz, cropy, cropx):
    z, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    startz = z // 2 - (cropz // 2)
    return img[startz:startz + cropz, starty:starty + cropy, startx:startx + cropx]


def center_pad_img(img, cropz, cropy, cropx):
    z, y, x = img.shape
    x_new = cropx if x < cropx else x
    y_new = cropy if y < cropy else y
    z_new = cropz if z < cropz else z
    p_x = (x_new - x) // 2
    p_y = (y_new - y) // 2
    p_z = (z_new - z) // 2
    return np.pad(img, pad_width=((p_z, z_new - z - p_z), (p_y, y_new - y - p_y), (p_x, x_new - x - p_x)),
                  mode='constant', constant_values=0)

def fix_labels(img,labels):
    mask = np.isin(img,labels)
    img[~mask] = 0.0
    for i in range(len(labels)):
        img[img==labels[i]] = i
    return img

def rename_labels(img,labels_map):
    for i in range(len(labels_map)):
        img[img==i] = labels_map[i]
    return img

def one_hot(a, num_classes):
    a = a.astype(int)
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def get_dataset(DATA_INFO, train=True, valid=False, test=False):
    if DATA_INFO['DATA'] == 'ADNI':
        return ADNIDataset(DATA_INFO, train, valid, test)

class ADNIDataset(Dataset):

    def __init__(self, DATA_INFO, train, valid, test):

        #self.labels = np.load('../Data_Info/Labels_Common.npy')
        #self.labels_map = np.loadtxt('../Data_Info/Labels_Map.txt')
        self.ADNI_metadata = pd.read_pickle('../Data_Info/ADNI1_enc.pkl').reset_index(drop=True)
        self.ADNI_metadata = self.ADNI_metadata.sort_values(by=['Subject ID'])
        #self.ADNI_metadata = self.ADNI_metadata.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.mri = self.ADNI_metadata["T1"]
        self.sex = self.ADNI_metadata["Sex"].values
        self.age = self.ADNI_metadata["Age"].values
        self.Pulse_Sequence = self.ADNI_metadata["Pulse_Sequence"].values
        self.Coil = self.ADNI_metadata["Coil"].values
        self.Acquisition_Plane = self.ADNI_metadata["Acquisition_Plane"].values
        self.Manufacturer = self.ADNI_metadata["Manufacturer"].values
        self.Mfg_Model = self.ADNI_metadata["Mfg_Model"].values
        self.Field_Strength = self.ADNI_metadata["Field_Strength"].values
        self.TE = self.ADNI_metadata["TE"].values
        self.TR = self.ADNI_metadata["TR"].values
        self.TI = self.ADNI_metadata["TI"].values
        self.Flip_Angle = self.ADNI_metadata["Flip_Angle"].values
        self.Matrix_X = self.ADNI_metadata["Matrix_X"].values
        self.Matrix_Y = self.ADNI_metadata["Matrix_Y"].values
        self.Matrix_Z = self.ADNI_metadata["Matrix_Z"].values
        self.Slice_Thickness = self.ADNI_metadata["Slice_Thickness"].values
        self.Pixel_Spacing_X = self.ADNI_metadata["Pixel_Spacing_X"].values
        self.Pixel_Spacing_Y = self.ADNI_metadata["Pixel_Spacing_Y"].values
        self.CSF_mean_intensity = self.ADNI_metadata["CSF_mean_intensity"].values
        self.GM_mean_intensity = self.ADNI_metadata["GM_mean_intensity"].values
        self.WM_mean_intensity = self.ADNI_metadata["WM_mean_intensity"].values
        self.seg = self.ADNI_metadata["Seg"]
        self.group = self.ADNI_metadata["Group"].values
        self.mmse = self.ADNI_metadata["MMSE Total Score"].values
        self.version = self.ADNI_metadata["Version"].values
        self.transforms = DATA_INFO['TRANSFORMS']


        n_samples = len(self.version)
        indices = np.arange(n_samples)
        train_idxs, valid_idxs, y_train, y_valid = train_test_split(indices, self.group.astype(int), test_size=0.2, random_state=seed)
        group_train = self.group[train_idxs].astype(int)
        train_idxs_0 = train_idxs[group_train == 0]
        train_idxs_1 = train_idxs[group_train == 1]
        diff = len(train_idxs_0) - len(train_idxs_1)
        added_idxs = train_idxs_1[np.random.choice(len(train_idxs_1), diff)]
        train_idxs = np.hstack((train_idxs,added_idxs))  

        if train:
            self.mri = self.mri[train_idxs].tolist()
            self.seg = self.seg[train_idxs].tolist()
            self.age = self.age[train_idxs]
            self.sex = self.sex[train_idxs]
            self.Pulse_Sequence = self.Pulse_Sequence[train_idxs]
            self.Coil = self.Coil[train_idxs]
            self.Acquisition_Plane = self.Acquisition_Plane[train_idxs]
            self.Manufacturer = self.Manufacturer[train_idxs]
            self.Mfg_Model = self.Mfg_Model[train_idxs]
            self.Field_Strength = self.Field_Strength[train_idxs]
            self.TE = self.TE[train_idxs]
            self.TR = self.TR[train_idxs]
            self.TI = self.TI[train_idxs]
            self.Flip_Angle = self.Flip_Angle[train_idxs]
            self.Matrix_X = self.Matrix_X[train_idxs]
            self.Matrix_Y = self.Matrix_Y[train_idxs]
            self.Matrix_Z = self.Matrix_Z[train_idxs]
            self.Slice_Thickness = self.Slice_Thickness[train_idxs]
            self.Pixel_Spacing_X = self.Pixel_Spacing_X[train_idxs]
            self.Pixel_Spacing_Y = self.Pixel_Spacing_Y[train_idxs]
            self.CSF_mean_intensity = self.CSF_mean_intensity[train_idxs]
            self.GM_mean_intensity = self.GM_mean_intensity[train_idxs]
            self.WM_mean_intensity = self.WM_mean_intensity[train_idxs]
            self.group = self.group[train_idxs]
            self.mmse = self.mmse[train_idxs]
            print(len(self.mri))
            assert len(self.mri) == len(self.seg)

        if valid:
            self.mri = self.mri[valid_idxs].tolist()
            self.seg = self.seg[valid_idxs].tolist()
            self.age = self.age[valid_idxs]
            self.sex = self.sex[valid_idxs]
            self.Pulse_Sequence = self.Pulse_Sequence[valid_idxs]
            self.Coil = self.Coil[valid_idxs]
            self.Acquisition_Plane = self.Acquisition_Plane[valid_idxs]
            self.Manufacturer = self.Manufacturer[valid_idxs]
            self.Mfg_Model = self.Mfg_Model[valid_idxs]
            self.Field_Strength = self.Field_Strength[valid_idxs]
            self.TE = self.TE[valid_idxs]
            self.TR = self.TR[valid_idxs]
            self.TI = self.TI[valid_idxs]
            self.Flip_Angle = self.Flip_Angle[valid_idxs]
            self.Matrix_X = self.Matrix_X[valid_idxs]
            self.Matrix_Y = self.Matrix_Y[valid_idxs]
            self.Matrix_Z = self.Matrix_Z[valid_idxs]
            self.Slice_Thickness = self.Slice_Thickness[valid_idxs]
            self.Pixel_Spacing_X = self.Pixel_Spacing_X[valid_idxs]
            self.Pixel_Spacing_Y = self.Pixel_Spacing_Y[valid_idxs]
            self.CSF_mean_intensity = self.CSF_mean_intensity[valid_idxs]
            self.GM_mean_intensity = self.GM_mean_intensity[valid_idxs]
            self.WM_mean_intensity = self.WM_mean_intensity[valid_idxs]
            self.group = self.group[valid_idxs]
            self.mmse = self.mmse[valid_idxs]
            print(len(self.mri))
            assert len(self.mri) == len(self.seg)

        if test:
            test_idxs = valid_idxs
            self.mri = self.mri[test_idxs].tolist()
            self.seg = self.seg[test_idxs].tolist()
            self.age = self.age[test_idxs]
            self.sex = self.sex[test_idxs]
            self.Pulse_Sequence = self.Pulse_Sequence[test_idxs]
            self.Coil = self.Coil[test_idxs]
            self.Acquisition_Plane = self.Acquisition_Plane[test_idxs]
            self.Manufacturer = self.Manufacturer[test_idxs]
            self.Mfg_Model = self.Mfg_Model[test_idxs]
            self.Field_Strength = self.Field_Strength[test_idxs]
            self.TE = self.TE[test_idxs]
            self.TR = self.TR[test_idxs]
            self.TI = self.TI[test_idxs]
            self.Flip_Angle = self.Flip_Angle[test_idxs]
            self.Matrix_X = self.Matrix_X[test_idxs]
            self.Matrix_Y = self.Matrix_Y[test_idxs]
            self.Matrix_Z = self.Matrix_Z[test_idxs]
            self.Slice_Thickness = self.Slice_Thickness[test_idxs]
            self.Pixel_Spacing_X = self.Pixel_Spacing_X[test_idxs]
            self.Pixel_Spacing_Y = self.Pixel_Spacing_Y[test_idxs]
            self.CSF_mean_intensity = self.CSF_mean_intensity[test_idxs]
            self.GM_mean_intensity = self.GM_mean_intensity[test_idxs]
            self.WM_mean_intensity = self.WM_mean_intensity[test_idxs]
            self.group = self.group[test_idxs]
            self.mmse = self.mmse[test_idxs]
            print(len(self.mri))
            assert len(self.mri) == len(self.seg)


    def __len__(self):
        return len(self.mri)

    def __getitem__(self, idx):

        #Data_Folder = '/proj/NIRAL/users/mahmoud/Data/ADNI/'
        Data_Folder = '/work/MultiTaskLearningProject/Data/ADNI1_CN_Vs_AD/'

        p = Path(self.mri[idx])
        mri_merged_sub = os.path.join(Path(*p.parts[:-1]),'brain_crop_sub.nrrd')
        mri = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(Data_Folder, mri_merged_sub),sitk.sitkFloat32)).astype('float32')
        mri = center_pad_img(mri, 128, 96, 96)
        mri = center_crop_img(mri, 128, 96, 96)
        #mri = mri[::2, ::2, ::2]
        mri = np.expand_dims(mri, axis=0)

        p = Path(self.seg[idx])
        seg_merged_sub = os.path.join(Path(*p.parts[:-1]),'aseg_crop_merged_sub_fix.nrrd')
        seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(Data_Folder, seg_merged_sub),sitk.sitkFloat32)).astype('float32')
        seg[seg==8] = 0;
        #seg = fix_labels(seg,self.labels)
        #seg = rename_labels(seg,self.labels_map)
        seg = center_pad_img(seg, 128, 96, 96)
        seg = center_crop_img(seg, 128, 96, 96)
        #seg = seg[::2, ::2, ::2]
        seg = np.expand_dims(seg, axis=0)

        if self.transforms is not None:
            mri, seg = self.transforms(mri, seg)

        age = self.age[idx]
        sex = self.sex[idx]
        Pulse_Sequence = self.Pulse_Sequence[idx]
        Coil = self.Coil[idx]
        #Acquisition_Plane = np.asarray(self.Acquisition_Plane[idx]).astype('int')
        Manufacturer = self.Manufacturer[idx]
        Mfg_Model = self.Mfg_Model[idx]
        Field_Strength = self.Field_Strength[idx]
        Pulse_Sequence = np.asarray(self.Pulse_Sequence[idx]).astype('float32')
        TE = self.TE[idx]
        TR = self.TR[idx]
        TI = self.TI[idx]
        Flip_Angle = self.Flip_Angle[idx]
        Matrix_X = self.Matrix_X[idx]
        Matrix_Y = self.Matrix_Y[idx]
        Matrix_Z = self.Matrix_Z[idx]
        #Slice_Thickness = np.asarray(self.Slice_Thickness[idx]).astype('float32')
        Pixel_Spacing_X = self.Pixel_Spacing_X[idx]
        Pixel_Spacing_Y = self.Pixel_Spacing_Y[idx]
        CSF_mean_intensity = self.CSF_mean_intensity[idx]
        GM_mean_intensity = self.GM_mean_intensity[idx]
        WM_mean_intensity = self.WM_mean_intensity[idx]

        meta_cont = np.hstack((TE,TR,TI,Flip_Angle,Matrix_X,Matrix_Y,Matrix_Z,Pixel_Spacing_X,Pixel_Spacing_Y,age,
            CSF_mean_intensity, GM_mean_intensity, WM_mean_intensity))

        meta_disc = np.hstack((one_hot(Pulse_Sequence,4),one_hot(Coil,9), one_hot(Manufacturer,3),one_hot(Mfg_Model,19),
            one_hot(Field_Strength,2),one_hot(sex,2)))

        meta = np.hstack((meta_disc,meta_cont)).astype('float32')

        group = np.asarray(self.group[idx]).astype('float32')
        group = np.expand_dims(group, axis=0)

        mmse = np.asarray(self.mmse[idx]).astype('float32')

        if mmse < 10.0:
            mmse = 10.0
        mmse = (mmse - 10.0) / 20.0
        mmse = (np.exp(mmse) - 1) / (2.7183 - 1)
        mmse = np.expand_dims(np.asarray(mmse).astype('float32'), axis=0)

        return TensorList(torch.from_numpy(mri), torch.from_numpy(meta)), [torch.from_numpy(seg),
                                                                           torch.from_numpy(group),
                                                                           torch.from_numpy(mmse)]
