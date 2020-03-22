import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import callbacks 
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Flatten, Activation, Dropout, Input, Concatenate
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix 
from scipy import interp
import time
from keras.optimizers import SGD
from keras.constraints import maxnorm
import time
from scipy.io import loadmat
import scipy.io
from keras import backend as K

 
############################################## Customized Classes #####################################################

#-------------------------------------------------------------------------------------------------------------
class UpdateRegions(Layer):
    def __init__(self, indices,no_of_vertices, no_of_regions, no_of_measurements,batch_size, **kwargs):
        self.indices = indices
        self.no_of_vertices = no_of_vertices
        self.no_of_regions = no_of_regions
        self.no_of_measurements = no_of_measurements
        self.batch_size = batch_size
        super(UpdateRegions, self).__init__(**kwargs)
    def build(self, input_shape):
        super(UpdateRegions, self).build(input_shape)
    def call(self, x):
        x  = K.reshape(x, shape=(self.no_of_vertices,))
        x  = K.gather(x,self.indices)
        x  = K.reshape(x, shape=(self.batch_size,self.no_of_vertices,self.no_of_regions,self.no_of_measurements))
        x  = K.mean(x, axis = -1)
        return x
    def compute_output_shape(self, input_shape):
        return (None, self.no_of_vertices, self.no_of_regions)

#-------------------------------------------------------------------------------------------------------------

class GeodesicConvolution(Layer):
    def __init__(self, nb_filters, **kwargs):
        self.nb_filters = nb_filters
        super(GeodesicConvolution, self).__init__(**kwargs)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(input_shape[2], self.nb_filters),
                                      initializer='uniform',
                                      trainable=True,name = 'kernel')
        super(GeodesicConvolution, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x):
        return K.dot(x,self.kernel)
    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.nb_filters)

#-------------------------------------------------------------------------------------------------------------

class MaxPoolFeatures(Layer):
    def __init__(self, **kwargs):
        super(MaxPoolFeatures, self).__init__(**kwargs)
    def build(self, input_shape):
        super(MaxPoolFeatures, self).build(input_shape)  
    def call(self, x):
        return K.max(x, axis=2, keepdims=False)
    def compute_output_shape(self, input_shape):
        return (None, input_shape[1])

#-------------------------------------------------------------------------------------------------------------

class ResampleSurface(Layer):
    def __init__(self, in_dim, out_dim, batch_size, ResampleMap, no_of_neighbours, **kwargs):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.ResampleMap = ResampleMap
        self.no_of_neighbours = no_of_neighbours
        super(ResampleSurface, self).__init__(**kwargs)
    def build(self, input_shape):
        super(ResampleSurface, self).build(input_shape)  
    def call(self, x):
        x  = K.reshape(x, shape=(self.in_dim,))
        x  = K.gather(x,self.ResampleMap)
        x  = K.reshape(x, shape=(self.batch_size,self.out_dim,self.no_of_neighbours))
        x  = K.mean(x, axis = -1)
        return x
    def compute_output_shape(self, input_shape):
        return (None, self.out_dim)


        #-------------------------------------------------------------------------------------------------------------

class MaskSurface(Layer):
    def __init__(self, in_dim, out_dim, batch_size, Mask, **kwargs):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.Mask = Mask
        super(MaskSurface, self).__init__(**kwargs)
    def build(self, input_shape):
        super(MaskSurface, self).build(input_shape)  
    def call(self, x):
        x  = K.reshape(x, shape=(self.in_dim,))
        x  = K.gather(x,self.Mask)
        x  = K.reshape(x, shape=(self.batch_size,self.out_dim))
        return x
    def compute_output_shape(self, input_shape):
        return (None, self.out_dim)

############################################## Read Data #############################################################
Input_no_of_vertices_LH = 40962 
Input_no_of_vertices_RH = 40962 
no_of_regions = 7
no_of_measurements = 6
no_of_neighbours = 5

L1_Input_no_of_vertices_LH = 20482 
L2_Input_no_of_vertices_LH = 10242 
L3_Input_no_of_vertices_LH = 5122 
L4_Input_no_of_vertices_LH = 2562 
L5_Input_no_of_vertices_LH = 1282 
L6_Input_no_of_vertices_LH = 642
L7_Input_no_of_vertices_LH = 322 
L8_Input_no_of_vertices_LH = 162

L1_Input_no_of_vertices_RH = 20482 
L2_Input_no_of_vertices_RH = 10242 
L3_Input_no_of_vertices_RH = 5122 
L4_Input_no_of_vertices_RH = 2562 
L5_Input_no_of_vertices_RH = 1282 
L6_Input_no_of_vertices_RH = 642
L7_Input_no_of_vertices_RH = 322 
L8_Input_no_of_vertices_RH = 162

no_of_subjects = 86
no_of_features = Input_no_of_vertices_LH

#Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#Read Data
dataset = loadmat('All_area_CN_Vs_AD.mat')['Data']
dataset = np.asarray(dataset)

# split into input (X) and output (Y) variables
X = dataset[0:no_of_subjects,0:2*no_of_features].astype(float)
Gendar = dataset[0:no_of_subjects,2*no_of_features].astype(float)
Gendar = np.reshape(Gendar, (no_of_subjects, 1))
Age = dataset[0:no_of_subjects,2*no_of_features+1].astype(float)
Age = np.reshape(Age, (no_of_subjects, 1))
X = np.concatenate((X,Gendar,Age), axis=1)
Y = dataset[0:no_of_subjects,2*no_of_features+2]
ID = dataset[0:no_of_subjects,2*no_of_features+3]
Fold_Num = dataset[0:no_of_subjects,2*no_of_features+4]

#Preprocess Data
X = preprocessing.scale(X)
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

#Read Grid Regions (LH)
Regions_LH_HR = pd.read_csv('./LH_Grid/Avg_LH_GM_Regions_Ordered.csv', header=None).values
Regions_LH_HR = Regions_LH_HR[0:Input_no_of_vertices_LH*no_of_regions,1:no_of_measurements+2]
Regions_LH_HR = Regions_LH_HR.flatten().astype(int)
Regions_LH_L1 = pd.read_csv('./LH_Grid/Deciamted_L1_Regions_Ordered.csv', header=None).values
Regions_LH_L1 = Regions_LH_L1[0:L1_Input_no_of_vertices_LH*no_of_regions,1:no_of_measurements+2]
Regions_LH_L1 = Regions_LH_L1.flatten().astype(int)
Regions_LH_L2 = pd.read_csv('./LH_Grid/Deciamted_L2_Regions_Ordered.csv', header=None).values
Regions_LH_L2 = Regions_LH_L2[0:L2_Input_no_of_vertices_LH*no_of_regions,1:no_of_measurements+2]
Regions_LH_L2 = Regions_LH_L2.flatten().astype(int)
Regions_LH_L3 = pd.read_csv('./LH_Grid/Deciamted_L3_Regions_Ordered.csv', header=None).values
Regions_LH_L3 = Regions_LH_L3[0:L3_Input_no_of_vertices_LH*no_of_regions,1:no_of_measurements+2]
Regions_LH_L3 = Regions_LH_L3.flatten().astype(int)
Regions_LH_L4 = pd.read_csv('./LH_Grid/Deciamted_L4_Regions_Ordered.csv', header=None).values
Regions_LH_L4 = Regions_LH_L4[0:L4_Input_no_of_vertices_LH*no_of_regions,1:no_of_measurements+2]
Regions_LH_L4 = Regions_LH_L4.flatten().astype(int)
Regions_LH_L5 = pd.read_csv('./LH_Grid/Deciamted_L5_Regions_Ordered.csv', header=None).values
Regions_LH_L5 = Regions_LH_L5[0:L5_Input_no_of_vertices_LH*no_of_regions,1:no_of_measurements+2]
Regions_LH_L5 = Regions_LH_L5.flatten().astype(int)
Regions_LH_L6 = pd.read_csv('./LH_Grid/Deciamted_L6_Regions_Ordered.csv', header=None).values
Regions_LH_L6 = Regions_LH_L6[0:L6_Input_no_of_vertices_LH*no_of_regions,1:no_of_measurements+2]
Regions_LH_L6 = Regions_LH_L6.flatten().astype(int)
Regions_LH_L7 = pd.read_csv('./LH_Grid/Deciamted_L7_Regions_Ordered.csv', header=None).values
Regions_LH_L7 = Regions_LH_L7[0:L7_Input_no_of_vertices_LH*no_of_regions,1:no_of_measurements+2]
Regions_LH_L7 = Regions_LH_L7.flatten().astype(int)
Regions_LH_L8 = pd.read_csv('./LH_Grid/Deciamted_L8_Regions_Ordered.csv', header=None).values
Regions_LH_L8 = Regions_LH_L8[0:L8_Input_no_of_vertices_LH*no_of_regions,1:no_of_measurements+2]
Regions_LH_L8 = Regions_LH_L8.flatten().astype(int)

#Read Decimation Maps (LH)
L1_LH_Map = pd.read_csv('./LH_Grid/DeciamtionMap_L1.csv', header=None).values
L1_LH_Map = L1_LH_Map[0:L1_Input_no_of_vertices_LH,:]
L1_LH_Map = L1_LH_Map.flatten().astype(int)
L2_LH_Map = pd.read_csv('./LH_Grid/DeciamtionMap_L2.csv', header=None).values
L2_LH_Map = L2_LH_Map[0:L2_Input_no_of_vertices_LH,:]
L2_LH_Map = L2_LH_Map.flatten().astype(int)
L3_LH_Map = pd.read_csv('./LH_Grid/DeciamtionMap_L3.csv', header=None).values
L3_LH_Map = L3_LH_Map[0:L3_Input_no_of_vertices_LH,:]
L3_LH_Map = L3_LH_Map.flatten().astype(int)
L4_LH_Map = pd.read_csv('./LH_Grid/DeciamtionMap_L4.csv', header=None).values
L4_LH_Map = L4_LH_Map[0:L4_Input_no_of_vertices_LH,:]
L4_LH_Map = L4_LH_Map.flatten().astype(int)
L5_LH_Map = pd.read_csv('./LH_Grid/DeciamtionMap_L5.csv', header=None).values
L5_LH_Map = L5_LH_Map[0:L5_Input_no_of_vertices_LH,:]
L5_LH_Map = L5_LH_Map.flatten().astype(int)
L6_LH_Map = pd.read_csv('./LH_Grid/DeciamtionMap_L6.csv', header=None).values
L6_LH_Map = L6_LH_Map[0:L6_Input_no_of_vertices_LH,:]
L6_LH_Map = L6_LH_Map.flatten().astype(int)
L7_LH_Map = pd.read_csv('./LH_Grid/DeciamtionMap_L7.csv', header=None).values
L7_LH_Map = L7_LH_Map[0:L7_Input_no_of_vertices_LH,:]
L7_LH_Map = L7_LH_Map.flatten().astype(int)
L8_LH_Map = pd.read_csv('./LH_Grid/DeciamtionMap_L8.csv', header=None).values
L8_LH_Map = L8_LH_Map[0:L8_Input_no_of_vertices_LH,:]
L8_LH_Map = L8_LH_Map.flatten().astype(int)

#Read Grid Regions (RH)
Regions_RH_HR = pd.read_csv('./RH_Grid/Avg_RH_GM_Regions_Ordered.csv', header=None).values
Regions_RH_HR = Regions_RH_HR[0:Input_no_of_vertices_RH*no_of_regions,1:no_of_measurements+2]
Regions_RH_HR = Regions_RH_HR.flatten().astype(int)
Regions_RH_L1 = pd.read_csv('./RH_Grid/Deciamted_L1_Regions_Ordered.csv', header=None).values
Regions_RH_L1 = Regions_RH_L1[0:L1_Input_no_of_vertices_RH*no_of_regions,1:no_of_measurements+2]
Regions_RH_L1 = Regions_RH_L1.flatten().astype(int)
Regions_RH_L2 = pd.read_csv('./RH_Grid/Deciamted_L2_Regions_Ordered.csv', header=None).values
Regions_RH_L2 = Regions_RH_L2[0:L2_Input_no_of_vertices_RH*no_of_regions,1:no_of_measurements+2]
Regions_RH_L2 = Regions_RH_L2.flatten().astype(int)
Regions_RH_L3 = pd.read_csv('./RH_Grid/Deciamted_L3_Regions_Ordered.csv', header=None).values
Regions_RH_L3 = Regions_RH_L3[0:L3_Input_no_of_vertices_RH*no_of_regions,1:no_of_measurements+2]
Regions_RH_L3 = Regions_RH_L3.flatten().astype(int)
Regions_RH_L4 = pd.read_csv('./RH_Grid/Deciamted_L4_Regions_Ordered.csv', header=None).values
Regions_RH_L4 = Regions_RH_L4[0:L4_Input_no_of_vertices_RH*no_of_regions,1:no_of_measurements+2]
Regions_RH_L4 = Regions_RH_L4.flatten().astype(int)
Regions_RH_L5 = pd.read_csv('./RH_Grid/Deciamted_L5_Regions_Ordered.csv', header=None).values
Regions_RH_L5 = Regions_RH_L5[0:L5_Input_no_of_vertices_RH*no_of_regions,1:no_of_measurements+2]
Regions_RH_L5 = Regions_RH_L5.flatten().astype(int)
Regions_RH_L6 = pd.read_csv('./RH_Grid/Deciamted_L6_Regions_Ordered.csv', header=None).values
Regions_RH_L6 = Regions_RH_L6[0:L6_Input_no_of_vertices_RH*no_of_regions,1:no_of_measurements+2]
Regions_RH_L6 = Regions_RH_L6.flatten().astype(int)
Regions_RH_L7 = pd.read_csv('./RH_Grid/Deciamted_L7_Regions_Ordered.csv', header=None).values
Regions_RH_L7 = Regions_RH_L7[0:L7_Input_no_of_vertices_RH*no_of_regions,1:no_of_measurements+2]
Regions_RH_L7 = Regions_RH_L7.flatten().astype(int)
Regions_RH_L8 = pd.read_csv('./RH_Grid/Deciamted_L8_Regions_Ordered.csv', header=None).values
Regions_RH_L8 = Regions_RH_L8[0:L8_Input_no_of_vertices_RH*no_of_regions,1:no_of_measurements+2]
Regions_RH_L8 = Regions_RH_L8.flatten().astype(int)

#Read Decimation Maps (RH)
L1_RH_Map = pd.read_csv('./RH_Grid/DeciamtionMap_L1.csv', header=None).values
L1_RH_Map = L1_RH_Map[0:L1_Input_no_of_vertices_RH,:]
L1_RH_Map = L1_RH_Map.flatten().astype(int)
L2_RH_Map = pd.read_csv('./RH_Grid/DeciamtionMap_L2.csv', header=None).values
L2_RH_Map = L2_RH_Map[0:L2_Input_no_of_vertices_RH,:]
L2_RH_Map = L2_RH_Map.flatten().astype(int)
L3_RH_Map = pd.read_csv('./RH_Grid/DeciamtionMap_L3.csv', header=None).values
L3_RH_Map = L3_RH_Map[0:L3_Input_no_of_vertices_RH,:]
L3_RH_Map = L3_RH_Map.flatten().astype(int)
L4_RH_Map = pd.read_csv('./RH_Grid/DeciamtionMap_L4.csv', header=None).values
L4_RH_Map = L4_RH_Map[0:L4_Input_no_of_vertices_RH,:]
L4_RH_Map = L4_RH_Map.flatten().astype(int)
L5_RH_Map = pd.read_csv('./RH_Grid/DeciamtionMap_L5.csv', header=None).values
L5_RH_Map = L5_RH_Map[0:L5_Input_no_of_vertices_RH,:]
L5_RH_Map = L5_RH_Map.flatten().astype(int)
L6_RH_Map = pd.read_csv('./RH_Grid/DeciamtionMap_L6.csv', header=None).values
L6_RH_Map = L6_RH_Map[0:L6_Input_no_of_vertices_RH,:]
L6_RH_Map = L6_RH_Map.flatten().astype(int)
L7_RH_Map = pd.read_csv('./RH_Grid/DeciamtionMap_L7.csv', header=None).values
L7_RH_Map = L7_RH_Map[0:L7_Input_no_of_vertices_RH,:]
L7_RH_Map = L7_RH_Map.flatten().astype(int)
L8_RH_Map = pd.read_csv('./RH_Grid/DeciamtionMap_L8.csv', header=None).values
L8_RH_Map = L8_RH_Map[0:L8_Input_no_of_vertices_RH,:]
L8_RH_Map = L8_RH_Map.flatten().astype(int)

############################################## NN Model #############################################################
nb_filters = 64
batch_size = 1
Dropout_rate = 0.2
nb_epochs = 1

# LH 
input1 = Input(shape=(Input_no_of_vertices_LH,))
#Block (1)
x101 = UpdateRegions(input_shape = (Input_no_of_vertices_LH,), indices = Regions_LH_HR, no_of_vertices = Input_no_of_vertices_LH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(input1)
x102 = GeodesicConvolution(nb_filters = nb_filters)(x101)
#x103 = Activation('relu')(x102)
x104 = MaxPoolFeatures()(x102)
x105 = ResampleSurface(input_shape = (Input_no_of_vertices_LH,),in_dim = Input_no_of_vertices_LH, out_dim = L1_Input_no_of_vertices_LH, batch_size = batch_size, ResampleMap = L1_LH_Map, no_of_neighbours = no_of_neighbours)(x104)
#Block (2)
#x106 = UpdateRegions(indices = Regions_LH_L1, no_of_vertices = L1_Input_no_of_vertices_LH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x105)
#x107 = GeodesicConvolution(nb_filters = nb_filters)(x106)
#x108 = Activation('relu')(x107)
#x109 = MaxPoolFeatures()(x108)
x110 = ResampleSurface(in_dim = L1_Input_no_of_vertices_LH, out_dim = L2_Input_no_of_vertices_LH, batch_size = batch_size, ResampleMap = L2_LH_Map, no_of_neighbours = no_of_neighbours)(x105)
#Block (3)
x111 = UpdateRegions(indices = Regions_LH_L2, no_of_vertices = L2_Input_no_of_vertices_LH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x110)
x112 = GeodesicConvolution(nb_filters = nb_filters)(x111)
#x113 = Activation('relu')(x112)
x114 = MaxPoolFeatures()(x112)
x115 = ResampleSurface(in_dim = L2_Input_no_of_vertices_LH, out_dim = L3_Input_no_of_vertices_LH, batch_size = batch_size, ResampleMap = L3_LH_Map, no_of_neighbours = no_of_neighbours)(x114)
#Block (4)
#x116 = UpdateRegions(indices = Regions_LH_L3, no_of_vertices = L3_Input_no_of_vertices_LH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x115)
#x117 = GeodesicConvolution(nb_filters = nb_filters)(x116)
#x118 = Activation('relu')(x117)
#x119 = MaxPoolFeatures()(x118)
x120 = ResampleSurface(in_dim = L3_Input_no_of_vertices_LH, out_dim = L4_Input_no_of_vertices_LH, batch_size = batch_size, ResampleMap = L4_LH_Map, no_of_neighbours = no_of_neighbours)(x115)
#Block (5)
x121 = UpdateRegions(indices = Regions_LH_L4, no_of_vertices = L4_Input_no_of_vertices_LH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x120)
x122 = GeodesicConvolution(nb_filters = nb_filters)(x121)
#x123 = Activation('relu')(x122)
x124 = MaxPoolFeatures()(x122)
x125 = ResampleSurface(in_dim = L4_Input_no_of_vertices_LH, out_dim = L5_Input_no_of_vertices_LH, batch_size = batch_size, ResampleMap = L5_LH_Map, no_of_neighbours = no_of_neighbours)(x124)
#Block (6)
#x126 = UpdateRegions(indices = Regions_LH_L5, no_of_vertices = L5_Input_no_of_vertices_LH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x125)
#x127 = GeodesicConvolution(nb_filters = nb_filters)(x126)
#x128 = Activation('relu')(x127)
#x129 = MaxPoolFeatures()(x128)
x130 = ResampleSurface(in_dim = L5_Input_no_of_vertices_LH, out_dim = L6_Input_no_of_vertices_LH, batch_size = batch_size, ResampleMap = L6_LH_Map, no_of_neighbours = no_of_neighbours)(x125)
#Block (7)
x131 = UpdateRegions(indices = Regions_LH_L6, no_of_vertices = L6_Input_no_of_vertices_LH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x130)
x132 = GeodesicConvolution(nb_filters = nb_filters)(x131)
#x133 = Activation('relu')(x132)
x134 = MaxPoolFeatures()(x132)
x135 = ResampleSurface(in_dim = L6_Input_no_of_vertices_LH, out_dim = L7_Input_no_of_vertices_LH, batch_size = batch_size, ResampleMap = L7_LH_Map, no_of_neighbours = no_of_neighbours)(x134)
#Block (8)
#x136 = UpdateRegions(indices = Regions_LH_L7, no_of_vertices = L7_Input_no_of_vertices_LH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x135)
#x137 = GeodesicConvolution(nb_filters = nb_filters)(x136)
#x138 = Activation('relu')(x137)
#x139 = MaxPoolFeatures()(x138)
x140 = ResampleSurface(in_dim = L7_Input_no_of_vertices_LH, out_dim = L8_Input_no_of_vertices_LH, batch_size = batch_size, ResampleMap = L8_LH_Map, no_of_neighbours = no_of_neighbours)(x135)

# RH 
input2 = Input(shape=(Input_no_of_vertices_RH,))
#Block (1)
x201 = UpdateRegions(input_shape = (Input_no_of_vertices_RH,), indices = Regions_RH_HR, no_of_vertices = Input_no_of_vertices_RH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(input2)
x202 = GeodesicConvolution(nb_filters = nb_filters)(x201)
#x203 = Activation('relu')(x202)
x204 = MaxPoolFeatures()(x202)
x205 = ResampleSurface(input_shape = (Input_no_of_vertices_RH,),in_dim = Input_no_of_vertices_RH, out_dim = L1_Input_no_of_vertices_RH, batch_size = batch_size, ResampleMap = L1_RH_Map, no_of_neighbours = no_of_neighbours)(x204)
#Block (2)
#x206 = UpdateRegions(indices = Regions_RH_L1, no_of_vertices = L1_Input_no_of_vertices_RH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x205)
#x207 = GeodesicConvolution(nb_filters = nb_filters)(x206)
#x208 = Activation('relu')(x207)
#x209 = MaxPoolFeatures()(x208)
x210 = ResampleSurface(in_dim = L1_Input_no_of_vertices_RH, out_dim = L2_Input_no_of_vertices_RH, batch_size = batch_size, ResampleMap = L2_RH_Map, no_of_neighbours = no_of_neighbours)(x205)
#Block (3)
x211 = UpdateRegions(indices = Regions_RH_L2, no_of_vertices = L2_Input_no_of_vertices_RH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x210)
x212 = GeodesicConvolution(nb_filters = nb_filters)(x211)
#x213 = Activation('relu')(x212)
x214 = MaxPoolFeatures()(x212)
x215 = ResampleSurface(in_dim = L2_Input_no_of_vertices_RH, out_dim = L3_Input_no_of_vertices_RH, batch_size = batch_size, ResampleMap = L3_RH_Map, no_of_neighbours = no_of_neighbours)(x214)
#Block (4)
#x216 = UpdateRegions(indices = Regions_RH_L3, no_of_vertices = L3_Input_no_of_vertices_RH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x215)
#x217 = GeodesicConvolution(nb_filters = nb_filters)(x216)
#x218 = Activation('relu')(x217)
#x219 = MaxPoolFeatures()(x218)
x220 = ResampleSurface(in_dim = L3_Input_no_of_vertices_RH, out_dim = L4_Input_no_of_vertices_RH, batch_size = batch_size, ResampleMap = L4_RH_Map, no_of_neighbours = no_of_neighbours)(x215)
#Block (5)
x221 = UpdateRegions(indices = Regions_RH_L4, no_of_vertices = L4_Input_no_of_vertices_RH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x220)
x222 = GeodesicConvolution(nb_filters = nb_filters)(x221)
#x223 = Activation('relu')(x222)
x224 = MaxPoolFeatures()(x222)
x225 = ResampleSurface(in_dim = L4_Input_no_of_vertices_RH, out_dim = L5_Input_no_of_vertices_RH, batch_size = batch_size, ResampleMap = L5_RH_Map, no_of_neighbours = no_of_neighbours)(x224)
#Block (6)
#x226 = UpdateRegions(indices = Regions_RH_L5, no_of_vertices = L5_Input_no_of_vertices_RH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x225)
#x227 = GeodesicConvolution(nb_filters = nb_filters)(x226)
#x228 = Activation('relu')(x227)
#x229 = MaxPoolFeatures()(x228)
x230 = ResampleSurface(in_dim = L5_Input_no_of_vertices_RH, out_dim = L6_Input_no_of_vertices_RH, batch_size = batch_size, ResampleMap = L6_RH_Map, no_of_neighbours = no_of_neighbours)(x225)
#Block (7)
x231 = UpdateRegions(indices = Regions_RH_L6, no_of_vertices = L6_Input_no_of_vertices_RH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x230)
x232 = GeodesicConvolution(nb_filters = nb_filters)(x231)
#x233 = Activation('relu')(x232)
x234 = MaxPoolFeatures()(x232)
x235 = ResampleSurface(in_dim = L6_Input_no_of_vertices_RH, out_dim = L7_Input_no_of_vertices_RH, batch_size = batch_size, ResampleMap = L7_RH_Map, no_of_neighbours = no_of_neighbours)(x234)
#Block (8)
#x236 = UpdateRegions(indices = Regions_RH_L7, no_of_vertices = L7_Input_no_of_vertices_RH, no_of_regions = no_of_regions, no_of_measurements= no_of_measurements, batch_size=batch_size)(x235)
#x237 = GeodesicConvolution(nb_filters = nb_filters)(x236)
#x238 = Activation('relu')(x237)
#x239 = MaxPoolFeatures()(x238)
x240 = ResampleSurface(in_dim = L7_Input_no_of_vertices_RH, out_dim = L8_Input_no_of_vertices_RH, batch_size = batch_size, ResampleMap = L8_RH_Map, no_of_neighbours = no_of_neighbours)(x235)

#Output
input3 = Input(shape=(1,))
input4 = Input(shape=(1,))
x41 = Concatenate(axis=-1)([x140, x240, input3, input4])
x42 = Dense(256,kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3))(x41)
x43 = Dropout(0.2)(x42)
x44 = Dense(128,kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3))(x43)
x45 = Dropout(0.2)(x44)
x46 = Dense(64,kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3))(x45)
x47 = Dropout(0.2)(x46)
out = Dense(2, kernel_initializer='normal', activation='softmax')(x47)

#Model
model = Model(inputs=[input1, input2, input3, input4], outputs=out)
model.summary()
############################################## Validation Analysis #############################################################

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 1000)
mean_PPV = 0.0
mean_NPV = 0.0
mean_Sen = 0.0
mean_Spe = 0.0
Result_00 = 0.0
Result_01 = 0.0
Result_10 = 0.0
Result_11 = 0.0

for fold in range(10):

    X_Test      = X[Fold_Num == fold+1]
    ID_Test     = ID[Fold_Num == fold+1]
    Y_Test_Orig      = Y[Fold_Num == fold+1]
    print(Y_Test_Orig)
    Y_Test      = np_utils.to_categorical(Y_Test_Orig)
    print(Y_Test)

    X_Train       = X[Fold_Num != fold+1]
    ID_Train      = ID[Fold_Num != fold+1]
    Y_Train  = Y[Fold_Num != fold+1]

    #Minority oversampling (Match Majority Class)
    sm = SMOTE(k_neighbors=3,kind='regular',random_state=int(seed))
    X_Train, Y_Train = sm.fit_sample(X_Train, Y_Train)
    Y_Train          = np_utils.to_categorical(Y_Train)

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.fit([X_Train[:,0:no_of_features],X_Train[:,no_of_features:2*no_of_features],X_Train[:,2*no_of_features],X_Train[:,2*no_of_features+1]], Y_Train, epochs=nb_epochs, batch_size=batch_size, verbose=2)
    model.save('./Models/Model'+str(fold+1)+'.hdf5')
    probas_ = model.predict([X_Test[:,0:no_of_features], X_Test[:,no_of_features:2*no_of_features],X_Test[:,2*no_of_features],X_Test[:,2*no_of_features+1]], batch_size=batch_size)
    print(probas_)
    Y_hat = np.squeeze(np.asarray(np.round(probas_[:,1])))
    print(Y_hat)
    results = confusion_matrix(Y_Test, Y_hat)
    print (results)
    Result_00 += results[0][0]
    Result_01 += results[0][1]
    Result_10 += results[1][0]
    Result_11 += results[1][1]

    fpr, tpr, thresholds = roc_curve(Y_Test_Orig, probas_[:,1])
    mean_tpr += interp(mean_fpr, fpr, tpr)

TN = Result_00
print ("TN %f" % TN)
FP = Result_01
print ("FP %f" % FP)
FN = Result_10
print ("FN %f" % FN)
TP = Result_11
print ("TP %f" % TP)

mean_PPV += (float(TP)/(TP + FP))
mean_NPV += (float(TN)/(TN + FN))
mean_Sen += (float(TP)/(TP + FN))
mean_Spe += (float(TN)/(TN + FP))
mean_Acc = 0.0
mean_Acc += (float(TN+TP)/(TN + FP + FN + TP))
mean_Acc *= 100
mean_PPV *= 100 
mean_NPV *= 100
mean_Sen *= 100
mean_Spe *= 100
print ("Mean Acc is %f" % mean_Acc)
print ("Mean PPV is %f" % mean_PPV)
print ("Mean NPV is %f" % mean_NPV)
print ("Mean Sen is %f" % mean_Sen)
print ("Mean Spe is %f" % mean_Spe)

mean_tpr /= 10
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print ("Mean AUC is %f" % mean_auc)









