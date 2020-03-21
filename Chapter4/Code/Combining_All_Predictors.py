#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import sys
import os
from scipy.io import loadmat,savemat
import pandas as pd
from sklearn import preprocessing   
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import layers,models,optimizers,regularizers,losses
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from itertools import product
from numpy.linalg import norm
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import pickle
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] ="0"

# In[ ]:


print(tf.__version__)
print(tf.keras.__version__)


# # Load Neuroimaging Data

# In[ ]:


Feature_Name_1 = 'SA'
Atlas_Name_1 = 'Lobes'
Feature_Name_2 = 'SCI'
Atlas_Name_2 = 'Destrieux'
Feature_Name_3 = 'EACSF'
Atlas_Name_3 = 'Func'
Feature_Name_4 = 'CT'
Atlas_Name_4 = 'AAL'


# In[ ]:


Feature_1 = np.asarray(loadmat('./FInalData_HR/Data_'+Feature_Name_1+'_'+Atlas_Name_1+'.mat')['Data']).astype(float)
Feature_2 = np.asarray(loadmat('./FInalData_HR/Data_'+Feature_Name_2+'_'+Atlas_Name_2+'.mat')['Data']).astype(float)
Feature_3 = np.asarray(loadmat('./FInalData_HR/Data_'+Feature_Name_3+'_'+Atlas_Name_3+'.mat')['Data']).astype(float)
Feature_4 = np.asarray(loadmat('./FInalData_HR/Data_'+Feature_Name_4+'_'+Atlas_Name_4+'.mat')['Data']).astype(float)
ICV  = np.asarray(loadmat('./FInalData_HR/Data_ICV.mat')['Data']).astype(float)
AGE  = np.asarray(loadmat('./FInalData_HR/Data_AGE.mat')['Data']).astype(float)
GROUP  = np.asarray(loadmat('./FInalData_HR/Data_GROUP.mat')['Data']).astype(float)
GENDER = np.asarray(loadmat('./FInalData_HR/Data_GENDER.mat')['Data']).astype(float)


# In[ ]:


ID = Feature_1[:,0].reshape((-1,1))
X1 = np.hstack((Feature_1[:,1:],ICV[:,1].reshape((-1,1)),AGE[:,1].reshape((-1,1)),GENDER[:,1].reshape((-1,1))))
X2 = np.hstack((Feature_2[:,1:],ICV[:,1].reshape((-1,1)),AGE[:,1].reshape((-1,1)),GENDER[:,1].reshape((-1,1))))
X3 = np.hstack((Feature_3[:,1:],ICV[:,1].reshape((-1,1)),AGE[:,1].reshape((-1,1)),GENDER[:,1].reshape((-1,1))))
X4 = np.hstack((Feature_4[:,1:],ICV[:,1].reshape((-1,1)),AGE[:,1].reshape((-1,1)),GENDER[:,1].reshape((-1,1))))
y = GROUP[:,1]

# # Preprocess Data

# In[ ]:

scaler1 = StandardScaler()
X1 = scaler1.fit_transform(X1)
scaler2 = StandardScaler()
X2 = scaler2.fit_transform(X2)
scaler3 = StandardScaler()
X3 = scaler3.fit_transform(X3)
scaler4 = StandardScaler()
X4 = scaler4.fit_transform(X4)
y = preprocessing.LabelEncoder().fit_transform(y)
print(X1.shape)
print(X2.shape)
print(X3.shape)
print(X4.shape)
print(y.shape)

# normalize a vector to have unit norm
def normalize(weights):
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result

def grid_search(GTs, Preds1, Preds2, Preds3, Preds4):  
    # define weights and threshold to consider
    T = np.linspace(0.0, 1.0, num=6)
    w = np.linspace(0.0, 1.0, num=11)
    best_T, best_ACC, best_PPV, best_SEN, best_NPV, best_SPC, best_weights = [], [], [], [], [], [], []
    for i in range(len(GTs)):
        print('Fold Number: %d' % (i))
        best_T_F, best_ACC_F, best_PPV_F, best_SEN_F, best_NPV_F, best_SPC_F, best_weights_F = 0.0,0.0,0.0,0.0,0.0,0.0,None
        ACC = 0.0; PPV = 0.0; SEN = 0.0; NPV = 0.0; SPC = 0.0
        for t in T:
            # iterate all possible combinations (cartesian product)
            for weights in product(w, repeat=4):    
                weights = normalize(weights)
                # evaluate weights
                Pred=weights[0]*Preds1[i]+weights[1]*Preds2[i]+weights[2]*Preds3[i]+weights[3]*Preds4[i]
                ACC=float(accuracy_score(GTs[i],Pred>t))
                PPV=float(precision_score(GTs[i],Pred>t))
                SEN=float(recall_score(GTs[i],Pred>t))
                NPV=float(precision_score(GTs[i],Pred>t, pos_label=0))
                SPC=float(recall_score(GTs[i],Pred>t, pos_label=0))
                #print('Fold %d model ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (i, ACC, SEN, SPC, PPV, NPV))
                if (PPV + 1.5*SEN + NPV + SPC + ACC) > (best_PPV_F + 1.5*best_SEN_F + best_NPV_F + best_SPC_F +best_ACC_F) :
                    best_ACC_F, best_PPV_F, best_SEN_F, best_NPV_F, best_SPC_F, best_T_F, best_weights_F = ACC, PPV, SEN, NPV, SPC, t, weights
                    print('best Fold %d model ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (i, best_ACC_F, best_SEN_F, best_SPC_F, best_PPV_F, best_NPV_F))
        best_T.append(best_T_F); best_ACC.append(best_ACC_F); best_PPV.append(best_PPV_F); best_SEN.append(best_SEN_F)
        best_NPV.append(best_NPV_F), best_SPC.append(best_SPC_F); best_weights.append(best_weights_F)
   
    ACC=0.0; PPV=0.0; SEN=0.0; NPV=0.0; SPC=0.0
    for i in range(len(GTs)):
        ACC+=best_ACC[i]; PPV+=best_PPV[i]; SEN+=best_SEN[i]; NPV+=best_NPV[i]; SPC+=best_SPC[i]
    ACC/=len(GTs); PPV/=len(GTs); SEN/=len(GTs); NPV/=len(GTs); SPC/=len(GTs)
    print('Final ensamble model ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (ACC, SEN, SPC, PPV, NPV))

    return best_weights, best_T

# In[ ]:

def fit_model(trainX, trainy, valX, valy, L1_nodes=256, L2_nodes=64, L3_nodes=0, activation='relu', BN=False,
                droprate=0.2, initializer='glorot_uniform', L2_penality='0.001', 
                loss='binary_crossentropy', optimizer='sgd', lr=0.01, m=0.95,
                batch_size=64, epochs=500, class_weight=[0.5,0.5], seed=1):

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)

    # cleanup
    tf.keras.backend.clear_session()

    num_features = trainX.shape[1]
    
    model = models.Sequential()
    
    # Block (1)
    model.add(layers.Dense(L1_nodes, kernel_initializer=initializer, kernel_regularizer=regularizers.l2(L2_penality), input_dim=num_features))
    if BN:     
        model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.Dropout(droprate, seed=seed))
        
    # Block (2)
    model.add(layers.Dense(L2_nodes, kernel_initializer=initializer, kernel_regularizer=regularizers.l2(L2_penality)))
    if BN:     
        model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.Dropout(droprate, seed=seed))

    if L3_nodes > 0:
        # Block (2)
        model.add(layers.Dense(L2_nodes, kernel_initializer=initializer, kernel_regularizer=regularizers.l2(L2_penality)))
        if BN:     
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        model.add(layers.Dropout(droprate, seed=seed))
    
    # Output
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    if optimizer == 'sgd':
      opt = optimizers.SGD(learning_rate=lr, momentum=m, nesterov=True)

    if optimizer == 'adam':
      opt = optimizers.Adam(learning_rate=lr)
            
    model.compile(loss = loss, optimizer = opt, metrics=['accuracy'])

    model.fit(trainX, trainy, validation_data=(valX, valy), epochs=epochs, class_weight=class_weights, verbose=0, callbacks=[es])

    return(model)

Best_GTs = []; Best_IDs = []; Best_Preds1 = [] ; Best_Preds2 = []; Best_Preds3 = [] ; Best_Preds4 = []
Best_Models1 = []; Best_Models2 = []; Best_Models3 = [] ; Best_Models4 = []
Best_ACC1 = 0 ; Best_SEN1 = 0 ; Best_SPC1 = 0 ; Best_PPV1 = 0 ; Best_NPV1 = 0
Best_ACC2 = 0 ; Best_SEN2 = 0 ; Best_SPC2 = 0 ; Best_PPV2 = 0 ; Best_NPV2 = 0
Best_ACC3 = 0 ; Best_SEN3 = 0 ; Best_SPC3 = 0 ; Best_PPV3 = 0 ; Best_NPV3 = 0
Best_ACC4 = 0 ; Best_SEN4 = 0 ; Best_SPC4 = 0 ; Best_PPV4 = 0 ; Best_NPV4 = 0
Best_Seed = 0

for counter in range(1):

    print('Counter Number: %d' % (counter+1))

    # Set a seed value
    if counter == 0:
        seed = 9659
    else:
        seed = random.randint(1, 10000)
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # In[ ]:
    nfolds = 10
    cv = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)

    GTs = []; IDs = [] ; Preds1 = [] ; Preds2 = [] ; Preds3 = [] ; Preds4 = [] 
    Models1 = []; Models2 = [] ; Models3 = [] ; Models4 = []
    ACC1 = 0 ; SEN1 = 0 ; SPC1 = 0 ; PPV1 = 0 ; NPV1 = 0
    ACC2 = 0 ; SEN2 = 0 ; SPC2 = 0 ; PPV2 = 0 ; NPV2 = 0
    ACC3 = 0 ; SEN3 = 0 ; SPC3 = 0 ; PPV3 = 0 ; NPV3 = 0
    ACC4 = 0 ; SEN4 = 0 ; SPC4 = 0 ; PPV4 = 0 ; NPV4 = 0

    Fold = 1
    for train_index, test_index in cv.split(X1, y):

        print('Fold Number: %d' % (Fold))

        X1_train, X1_test = X1[train_index], X1[test_index]
        X2_train, X2_test = X2[train_index], X2[test_index]
        X3_train, X3_test = X3[train_index], X3[test_index]
        X4_train, X4_test = X4[train_index], X4[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ID_train, ID_test = ID[train_index], ID[test_index]

        class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)

    #-----------------------------------------------------SA-------------------------------------------------------
        sm1 = SMOTENC(random_state=seed, sampling_strategy=1.0, k_neighbors=1, categorical_features=[X1.shape[1]-1])
        X1_train_res, y1_train_res = sm1.fit_resample(X1_train, y_train)

        model1 = fit_model(X1_train_res, y1_train_res, X1_test, y_test, L1_nodes=60, L2_nodes=120, L3_nodes=0,
                                activation='relu', BN=False, droprate=0.2, initializer='glorot_uniform', 
                                L2_penality=0.001, loss='binary_crossentropy', optimizer='sgd', 
                                lr=0.001, m=0.95, batch_size=64, epochs=1000, class_weight=class_weights, seed=seed)

        Pred1 = model1.predict(X1_test)

        ACC1+= accuracy_score(y_test,Pred1>0.5)
        SEN1+= recall_score(y_test,Pred1>0.5)
        SPC1+= recall_score(y_test,Pred1>0.5, pos_label=0)
        PPV1+= precision_score(y_test,Pred1>0.5)
        NPV1+= precision_score(y_test,Pred1>0.5, pos_label=0)

    #s-----------------------------------------------------SCI-------------------------------------------------------
        sm2 = SMOTENC(random_state=seed, sampling_strategy=1.0, k_neighbors=3, categorical_features=[X2.shape[1]-1])
        X2_train_res, y2_train_res = sm2.fit_resample(X2_train, y_train)

        model2 = fit_model(X2_train_res, y2_train_res, X2_test, y_test, L1_nodes=50, L2_nodes=200, L3_nodes=0,
                                activation='tanh', BN=False, droprate=0.2, initializer='glorot_uniform', 
                                L2_penality=0.01, loss='binary_crossentropy', optimizer='sgd', 
                                lr=0.001, m=0.99, batch_size=32, epochs=1000, class_weight=class_weights, seed=seed)

        Pred2 = model2.predict(X2_test)

        ACC2+= accuracy_score(y_test,Pred2>0.5)
        SEN2+= recall_score(y_test,Pred2>0.5)
        SPC2+= recall_score(y_test,Pred2>0.5, pos_label=0)
        PPV2+= precision_score(y_test,Pred2>0.5)
        NPV2+= precision_score(y_test,Pred2>0.5, pos_label=0)

    #-----------------------------------------------------EACSF-------------------------------------------------------
        sm3 = SMOTENC(random_state=seed, sampling_strategy=1.0, k_neighbors=3, categorical_features=[X3.shape[1]-1])
        X3_train_res, y3_train_res = sm3.fit_resample(X3_train, y_train)

        model3 = fit_model(X3_train_res, y3_train_res, X3_test, y_test, L1_nodes=200, L2_nodes=150, L3_nodes=0,
                                activation='relu', BN=False, droprate=0.3, initializer='he_uniform', 
                                L2_penality=0.01, loss='binary_crossentropy', optimizer='sgd', 
                                lr=0.001, m=0.95, batch_size=32, epochs=1000, class_weight=class_weights, seed=seed)
    
        Pred3 = model3.predict(X3_test)

        ACC3+= accuracy_score(y_test,Pred3>0.5)
        SEN3+= recall_score(y_test,Pred3>0.5)
        SPC3+= recall_score(y_test,Pred3>0.5, pos_label=0)
        PPV3+= precision_score(y_test,Pred3>0.5)
        NPV3+= precision_score(y_test,Pred3>0.5, pos_label=0)

    #s-----------------------------------------------------CT-------------------------------------------------------
        sm4 = SMOTENC(random_state=seed, sampling_strategy=1.0, k_neighbors=3, categorical_features=[X4.shape[1]-1])
        X4_train_res, y4_train_res = sm4.fit_resample(X4_train, y_train)

        model4 = fit_model(X4_train_res, y4_train_res, X4_test, y_test, L1_nodes=16, L2_nodes=64, L3_nodes=0,
                                activation='relu', BN=False, droprate=0.2, initializer='he_normal', 
                                L2_penality=0.01, loss='binary_crossentropy', optimizer='adam', 
                                lr=0.001, m=0.99, batch_size=32, epochs=1000, class_weight=class_weights, seed=seed)

        Pred4 = model4.predict(X4_test)

        ACC4+= accuracy_score(y_test,Pred4>0.5)
        SEN4+= recall_score(y_test,Pred4>0.5)
        SPC4+= recall_score(y_test,Pred4>0.5, pos_label=0)
        PPV4+= precision_score(y_test,Pred4>0.5)
        NPV4+= precision_score(y_test,Pred4>0.5, pos_label=0)

        GTs.append(y_test); IDs.append(ID_test)
        Preds1.append(Pred1) ; Preds2.append(Pred2); Preds3.append(Pred3) ; Preds4.append(Pred4)
        Models1.append(model1); Models2.append(model2) ; Models3.append(model3) ; Models4.append(model4)

        Fold+= 1

    ACC1/=nfolds ; SEN1/=nfolds ; SPC1/=nfolds ; PPV1/=nfolds ; NPV1/=nfolds
    ACC2/=nfolds ; SEN2/=nfolds ; SPC2/=nfolds ; PPV2/=nfolds ; NPV2/=nfolds
    ACC3/=nfolds ; SEN3/=nfolds ; SPC3/=nfolds ; PPV3/=nfolds ; NPV3/=nfolds
    ACC4/=nfolds ; SEN4/=nfolds ; SPC4/=nfolds ; PPV4/=nfolds ; NPV4/=nfolds

    print('seed number: %d' % (seed))
    print('model1 ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (ACC1, SEN1, SPC1, PPV1, NPV1))
    print('model2 ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (ACC2, SEN2, SPC2, PPV2, NPV2))
    print('model3 ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (ACC3, SEN3, SPC3, PPV3, NPV3))
    print('model4 ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (ACC4, SEN4, SPC4, PPV4, NPV4))

    if ((ACC1+ACC2+ACC3+ACC4)) > ((Best_ACC1+Best_ACC2+Best_ACC3+Best_ACC4)):
        Best_ACC1 = ACC1 ; Best_SEN1 = SEN1 ; Best_SPC1 = SPC1 ; Best_PPV1 = PPV1 ; Best_NPV1 = NPV1
        Best_ACC2 = ACC2 ; Best_SEN2 = SEN2 ; Best_SPC2 = SPC2 ; Best_PPV2 = PPV2 ; Best_NPV2 = NPV2
        Best_ACC3 = ACC3 ; Best_SEN3 = SEN3 ; Best_SPC3 = SPC3 ; Best_PPV3 = PPV3 ; Best_NPV3 = NPV3
        Best_ACC4 = ACC4 ; Best_SEN4 = SEN4 ; Best_SPC4 = SPC4 ; Best_PPV4 = PPV4 ; Best_NPV4 = NPV4
        Best_GTs = GTs; Best_IDs = IDs; Best_Preds1 = Preds1 ; Best_Preds2 = Preds2; Best_Preds3 = Preds3 ; Best_Preds4 = Preds4
        Best_Models1 = Models1; Best_Models2 = Models2; Best_Models3 = Models3; Best_Models4 = Models4
        Best_Seed = seed

        print('best seed number: %d' % (Best_Seed))
        print('model1 ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (Best_ACC1, Best_SEN1, Best_SPC1, Best_PPV1, Best_NPV1))
        print('model2 ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (Best_ACC2, Best_SEN2, Best_SPC2, Best_PPV2, Best_NPV2))
        print('model3 ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (Best_ACC3, Best_SEN3, Best_SPC3, Best_PPV3, Best_NPV3))
        print('model4 ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (Best_ACC4, Best_SEN4, Best_SPC4, Best_PPV4, Best_NPV4))

print('best seed number: %d' % (Best_Seed))
print('model1 ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (Best_ACC1, Best_SEN1, Best_SPC1, Best_PPV1, Best_NPV1))
print('model2 ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (Best_ACC2, Best_SEN2, Best_SPC2, Best_PPV2, Best_NPV2))
print('model3 ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (Best_ACC3, Best_SEN3, Best_SPC3, Best_PPV3, Best_NPV3))
print('model4 ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (Best_ACC4, Best_SEN4, Best_SPC4, Best_PPV4, Best_NPV4))

Best_weights, Best_T = grid_search(Best_GTs, Best_Preds1, Best_Preds2, Best_Preds3, Best_Preds4)

with open('Best_weights.pkl', 'wb') as f:
    pickle.dump(Best_weights, f)
    
with open('Best_T.pkl', 'wb') as f:
    pickle.dump(Best_T, f)


ACC=0.0; PPV=0.0; SEN=0.0; NPV=0.0; SPC=0.0; i=0
for train_index, test_index in cv.split(X1, y):

    print('Fold Number: %d' % (i+1))

    X1_train, X1_test = X1[train_index], X1[test_index]
    X2_train, X2_test = X2[train_index], X2[test_index]
    X3_train, X3_test = X3[train_index], X3[test_index]
    X4_train, X4_test = X4[train_index], X4[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pred1 = Best_Models1[i].predict(X1_test) ; pred2 = Best_Models2[i].predict(X2_test) 
    pred3 = Best_Models3[i].predict(X3_test) ; pred4 = Best_Models4[i].predict(X4_test)
    W = Best_weights[i] ; T = Best_T[i]
    pred  = (W[0]*pred1+W[1]*pred2+W[2]*pred3+W[3]*pred4) > T

    ACC+=float(accuracy_score(y_test,pred))
    PPV+=float(precision_score(y_test,pred))
    SEN+=float(recall_score(y_test,pred))
    NPV+=float(precision_score(y_test,pred, pos_label=0))
    SPC+=float(recall_score(y_test,pred, pos_label=0))

    Best_Models1[i].save('SA_'+str(i)+'.h5')
    Best_Models2[i].save('SCI_'+str(i)+'.h5')
    Best_Models3[i].save('EACSF_'+str(i)+'.h5')
    Best_Models4[i].save('CT_'+str(i)+'.h5')

    i+= 1

ACC/=nfolds; PPV/=nfolds; SEN/=nfolds; NPV/=nfolds; SPC/=nfolds
print('Final ensamble model ACC: %.3f, SEN: %.3f, SPC: %.3f, PPV: %.3f, NPV: %.3f' % (ACC, SEN, SPC, PPV, NPV))






