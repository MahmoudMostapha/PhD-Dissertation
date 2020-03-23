from datetime import datetime
import sys
import os
import argparse
from sklearn.model_selection import ParameterGrid
from scipy.io import loadmat,savemat
from sklearn import metrics
import tensorflow as tf
import numpy as np
from tensorflow.keras import utils
import pandas as pd
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pickle
import matplotlib.pyplot as pyplot
print(tf.VERSION)
print(tf.keras.__version__)
from skimage import io
from skimage.transform import resize
from OCNN import OC_NN
import keras


seed = 1000

def classify_using_ocnn(My_data_train, My_data_test,loadclassifier,modelname):
    insample_train = My_data_train
    insample_test, outsample_test = My_data_test

    # fit the model 
    if loadclassifier:
        DIM = insample_train.shape[1]
        HIDDEN =  np.load("./" + modelname + "_H.npy")
        NUM_EPOCHS = np.load("./" + modelname + "_NUM_EPOCHS.npy")
        BS = np.load("./" + modelname + "_BS.npy")
        nu =  np.load("./nu.npy")
        Activation_Unit = np.load("./Activation_Unit.npy")
        ocnn = OC_NN(DIM,HIDDEN,Activation_Unit,"./",modelname)
        SPC, Recall = ocnn.score(insample_test,outsample_test,nu)
        param = {"nu": nu, "HIDDEN": HIDDEN, "NUM_EPOCHS":NUM_EPOCHS,"BS":BS}
        if Recall > 0 and SPC < 1:
            Prev = float(outsample_test.shape[0]) / float(insample_test.shape[0])
            Precision = (Recall*Prev)/(Recall*Prev + (1-SPC)*(1-Prev))
            Accuracy  = Recall*Prev + SPC*(1-Prev)
            print('Prev : %f'%Prev)
            print('Recall : %f'%Recall)
            print('Precision : %f'%Precision)
            print('SPC : %f'%SPC)
            print('Accuracy : %f'%Accuracy)
            results = []
            Avg_score         = (Recall + Precision)/2
            results.append((param,Accuracy,Recall,Precision,SPC,Avg_score))
        else:
            results.append((param,0,0,0,0,0))
    else:

        grid = ParameterGrid({"nu": [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,.45,0.5],
                          "HIDDEN": [128,256,512],
                          "NUM_EPOCHS": [50],
                          "BS": [32],
                          "Activation_Unit": ["sigmoid", "linear"] })
        results = []
        for param in grid:
            print('=========  ',param)
            DIM = insample_train.shape[1]
            ocnn = OC_NN(DIM,param['HIDDEN'],param['Activation_Unit'],"./",modelname)
            ocnn.fit(insample_train,param['nu'],param['NUM_EPOCHS'],param['BS'])
            SPC, Recall = ocnn.score(insample_test,outsample_test,param['nu'])
            if Recall > 0 and SPC < 1:
                Prev = float(outsample_test.shape[0]) / float(insample_test.shape[0])
                Precision = (Recall*Prev)/(Recall*Prev + (1-SPC)*(1-Prev))
                Accuracy  = Recall*Prev + SPC*(1-Prev)
                print('Prev : %f'%Prev)
                print('Recall : %f'%Recall)
                print('Precision : %f'%Precision)
                print('SPC : %f'%SPC)
                print('Accuracy : %f'%Accuracy)
                results = []
                Avg_score         = (Recall + Precision)/2
                results.append((param,Accuracy,Recall,Precision,SPC,Avg_score))
            else:
                results.append((param,0,0,0,0,0))

        results.sort(key=lambda k: k[-1], reverse = True)
        print(results[0])

        DIM = insample_train.shape[1]
        np.save("./" + modelname + "_H.npy",results[0][0]['HIDDEN'])
        np.save("./" + modelname + "_NUM_EPOCHS.npy",results[0][0]['NUM_EPOCHS'])
        np.save("./" + modelname + "_BS.npy",results[0][0]['BS'])
        np.save("./nu.npy",results[0][0]['nu'])
        np.save("./Activation_Unit.npy",results[0][0]['Activation_Unit'])
        ocnn = OC_NN(DIM,results[0][0]['HIDDEN'],results[0][0]['Activation_Unit'],"./",modelname)
        ocnn.fit(insample_train,results[0][0]['nu'],results[0][0]['NUM_EPOCHS'],results[0][0]['BS'])

    return results[0][2],results[0][3],results

def classify_using_localoutlierfactor(My_data_train, My_data_test,loadclassifier,modelname):
    insample_train = My_data_train
    insample_test, outsample_test = My_data_test

    # fit the model 
    if loadclassifier:
        localoutlierfactor = pickle.load(open(modelname + '_localoutlierfactor_model.sav', 'rb'))
        results = []
        param = localoutlierfactor.get_params()
        y_insample       = localoutlierfactor.predict(insample_test)
        correct_insample = y_insample[y_insample == 1].size
        SPC   = correct_insample/y_insample.size
        y_outsample       = localoutlierfactor.predict(outsample_test)
        correct_outsample = y_outsample[y_outsample == -1].size
        Recall   = correct_outsample/y_outsample.size
        if Recall > 0 and SPC < 1:
            Prev = float(outsample_test.shape[0]) / float(insample_test.shape[0])
            Precision = (Recall*Prev)/(Recall*Prev + (1-SPC)*(1-Prev))
            Accuracy  = Recall*Prev + SPC*(1-Prev)
            print('Prev : %f'%Prev)
            print('Recall : %f'%Recall)
            print('Precision : %f'%Precision)
            print('SPC : %f'%SPC)
            print('Accuracy : %f'%Accuracy)
            Avg_score         = (Recall + Precision)/2
            results.append((param,Accuracy,Recall,Precision,SPC,Avg_score))
        else:
            results.append((param,0,0,0,0,0))

    else:

        grid = ParameterGrid({"contamination": [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,.45,0.5],
                          "metric": ['minkowski','cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
                          "novelty": [True]})
        results = []
        for param in grid:
            print('=========  ',param)
            localoutlierfactor = LocalOutlierFactor()
            localoutlierfactor.set_params(**param)
            localoutlierfactor.fit(insample_train)
            y_insample       = localoutlierfactor.predict(insample_test)
            correct_insample = y_insample[y_insample == 1].size
            SPC   = correct_insample/y_insample.size
            y_outsample       = localoutlierfactor.predict(outsample_test)
            correct_outsample = y_outsample[y_outsample == -1].size
            Recall   = correct_outsample/y_outsample.size
            if Recall > 0 and SPC < 1:
                Prev = float(outsample_test.shape[0]) / float(insample_test.shape[0])
                Precision = (Recall*Prev)/(Recall*Prev + (1-SPC)*(1-Prev))
                Accuracy  = Recall*Prev + SPC*(1-Prev)
                print('Prev : %f'%Prev)
                print('Recall : %f'%Recall)
                print('Precision : %f'%Precision)
                print('SPC : %f'%SPC)
                print('Accuracy : %f'%Accuracy)
                Avg_score         = (Recall + Precision)/2
                results.append((param,Accuracy,Recall,Precision,SPC,Avg_score))
            else:
                results.append((param,0,0,0,0,0))

        results.sort(key=lambda k: k[-1], reverse = True)
        print(results[0])

        localoutlierfactor = LocalOutlierFactor(**results[0][0])
        localoutlierfactor.fit(insample_train)
        pickle.dump(localoutlierfactor, open(modelname + '_localoutlierfactor_model.sav', 'wb'))

    return results[0][2],results[0][3],results

def classify_using_isolationforest(My_data_train, My_data_test,loadclassifier,modelname):
    insample_train = My_data_train
    insample_test, outsample_test = My_data_test

    # fit the model 
    if loadclassifier:
        isolationforest = pickle.load(open(modelname + '_isolationforest_model.sav', 'rb'))
        results = []
        param = isolationforest.get_params()
        y_insample       = isolationforest.predict(insample_test)
        correct_insample = y_insample[y_insample == 1].size
        SPC   = correct_insample/y_insample.size
        y_outsample       = isolationforest.predict(outsample_test)
        correct_outsample = y_outsample[y_outsample == -1].size
        Recall   = correct_outsample/y_outsample.size
        if Recall > 0 and SPC < 1:
            Prev = float(outsample_test.shape[0]) / float(insample_test.shape[0])
            Precision = (Recall*Prev)/(Recall*Prev + (1-SPC)*(1-Prev))
            Accuracy  = Recall*Prev + SPC*(1-Prev)
            print('Prev : %f'%Prev)
            print('Recall : %f'%Recall)
            print('Precision : %f'%Precision)
            print('SPC : %f'%SPC)
            print('Accuracy : %f'%Accuracy)
            Avg_score         = (Recall + Precision)/2
            results.append((param,Accuracy,Recall,Precision,SPC,Avg_score))
        else:
            results.append((param,0,0,0,0,0))

    else:

        grid = ParameterGrid({"contamination": [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,.45,0.5],
                          "behaviour": ['new'],
                          "random_state": [seed]})
        results = []
        for param in grid:
            print('=========  ',param)
            isolationforest = IsolationForest()
            isolationforest.set_params(**param)
            isolationforest.fit(insample_train)
            y_insample       = isolationforest.predict(insample_test)
            correct_insample = y_insample[y_insample == 1].size
            SPC   = correct_insample/y_insample.size
            y_outsample       = isolationforest.predict(outsample_test)
            correct_outsample = y_outsample[y_outsample == -1].size
            Recall   = correct_outsample/y_outsample.size
            if Recall > 0 and SPC < 1:
                Prev = float(outsample_test.shape[0]) / float(insample_test.shape[0])
                Precision = (Recall*Prev)/(Recall*Prev + (1-SPC)*(1-Prev))
                Accuracy  = Recall*Prev + SPC*(1-Prev)
                print('Prev : %f'%Prev)
                print('Recall : %f'%Recall)
                print('Precision : %f'%Precision)
                print('SPC : %f'%SPC)
                print('Accuracy : %f'%Accuracy)
                Avg_score         = (Recall + Precision)/2
                results.append((param,Accuracy,Recall,Precision,SPC,Avg_score))
            else:
                results.append((param,0,0,0,0,0))

        results.sort(key=lambda k: k[-1], reverse = True)
        print(results[0])

        isolationforest = IsolationForest(**results[0][0])
        isolationforest.fit(insample_train)
        pickle.dump(isolationforest, open(modelname + '_isolationforest_model.sav', 'wb'))

    return results[0][2],results[0][3],results

def classify_using_robustcovariance(My_data_train, My_data_test,loadclassifier,modelname):
    insample_train = My_data_train
    insample_test, outsample_test = My_data_test

    # fit the model 
    if loadclassifier:
        robustcovariance = pickle.load(open(modelname + '_robustcovariance_model.sav', 'rb'))
        results = []
        param = robustcovariance.get_params()
        y_insample       = robustcovariance.predict(insample_test)
        correct_insample = y_insample[y_insample == 1].size
        SPC   = correct_insample/y_insample.size
        y_outsample       = robustcovariance.predict(outsample_test)
        correct_outsample = y_outsample[y_outsample == -1].size
        Recall   = correct_outsample/y_outsample.size
        if Recall > 0 and SPC < 1:
            Prev = float(outsample_test.shape[0]) / float(insample_test.shape[0])
            Precision = (Recall*Prev)/(Recall*Prev + (1-SPC)*(1-Prev))
            Accuracy  = Recall*Prev + SPC*(1-Prev)
            print('Prev : %f'%Prev)
            print('Recall : %f'%Recall)
            print('Precision : %f'%Precision)
            print('SPC : %f'%SPC)
            print('Accuracy : %f'%Accuracy)
            Avg_score         = (Recall + Precision)/2
            results.append((param,Accuracy,Recall,Precision,SPC,Avg_score))
        else:
            results.append((param,0,0,0,0,0))

    else:

        grid = ParameterGrid({"contamination": [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,.45,0.5]})
        results = []
        for param in grid:
            print('=========  ',param)
            robustcovariance = EllipticEnvelope()
            robustcovariance.set_params(**param)
            robustcovariance.fit(insample_train)
            y_insample       = robustcovariance.predict(insample_test)
            correct_insample = y_insample[y_insample == 1].size
            SPC   = correct_insample/y_insample.size
            y_outsample       = robustcovariance.predict(outsample_test)
            correct_outsample = y_outsample[y_outsample == -1].size
            Recall   = correct_outsample/y_outsample.size
            if Recall > 0 and SPC < 1:
                Prev = float(outsample_test.shape[0]) / float(insample_test.shape[0])
                Precision = (Recall*Prev)/(Recall*Prev + (1-SPC)*(1-Prev))
                Accuracy  = Recall*Prev + SPC*(1-Prev)
                print('Prev : %f'%Prev)
                print('Recall : %f'%Recall)
                print('Precision : %f'%Precision)
                print('SPC : %f'%SPC)
                print('Accuracy : %f'%Accuracy)
                Avg_score         = (Recall + Precision)/2
                results.append((param,Accuracy,Recall,Precision,SPC,Avg_score))
            else:
                results.append((param,0,0,0,0,0))   

        results.sort(key=lambda k: k[-1], reverse = True)
        print(results[0])

        robustcovariance = EllipticEnvelope(**results[0][0])
        robustcovariance.fit(insample_train)
        pickle.dump(robustcovariance, open(modelname + '_robustcovariance_model.sav', 'wb'))

    return results[0][2],results[0][3],results

def classify_using_ocsvm(My_data_train, My_data_test,loadclassifier,modelname):
    insample_train = My_data_train
    insample_test, outsample_test = My_data_test

    # fit the model 
    if loadclassifier:
        ocsvm = pickle.load(open(modelname + '_ocsv_model.sav', 'rb'))
        results = []
        param = ocsvm.get_params()
        y_insample       = ocsvm.predict(insample_test)
        correct_insample = y_insample[y_insample == 1].size
        SPC   = correct_insample/y_insample.size
        y_outsample       = ocsvm.predict(outsample_test)
        correct_outsample = y_outsample[y_outsample == -1].size
        Recall   = correct_outsample/y_outsample.size
        if Recall > 0 and SPC < 1:
            Prev = float(outsample_test.shape[0]) / float(insample_test.shape[0])
            Precision = (Recall*Prev)/(Recall*Prev + (1-SPC)*(1-Prev))
            Accuracy  = Recall*Prev + SPC*(1-Prev)
            print('Prev : %f'%Prev)
            print('Recall : %f'%Recall)
            print('Precision : %f'%Precision)
            print('SPC : %f'%SPC)
            print('Accuracy : %f'%Accuracy)
            Avg_score         = (Recall + Precision)/2
            results.append((param,Accuracy,Recall,Precision,SPC,Avg_score))
        else:
            results.append((param,0,0,0,0,0))

            
    else:

        grid = ParameterGrid({"nu": [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,.45,0.5],
                          "kernel": ['linear', 'rbf'],
                          "gamma": ['auto', 'scale', 0.05, 0.1 , 0.2 , 0.3 ]})
        results = []
        for param in grid:
            print('=========  ',param)
            ocsvm = OneClassSVM()
            ocsvm.set_params(**param)
            ocsvm.fit(insample_train)
            y_insample       = ocsvm.predict(insample_test)
            correct_insample = y_insample[y_insample == 1].size
            SPC   = correct_insample/y_insample.size
            y_outsample       = ocsvm.predict(outsample_test)
            correct_outsample = y_outsample[y_outsample == -1].size
            Recall   = correct_outsample/y_outsample.size
            if Recall > 0 and SPC < 1:
                Prev = float(outsample_test.shape[0]) / float(insample_test.shape[0])
                Precision = (Recall*Prev)/(Recall*Prev + (1-SPC)*(1-Prev))
                Accuracy  = Recall*Prev + SPC*(1-Prev)
                print('Prev : %f'%Prev)
                print('Recall : %f'%Recall)
                print('Precision : %f'%Precision)
                print('SPC : %f'%SPC)
                print('Accuracy : %f'%Accuracy)
                Avg_score         = (Recall + Precision)/2
                results.append((param,Accuracy,Recall,Precision,SPC,Avg_score))
            else:
                results.append((param,0,0,0,0,0))

        results.sort(key=lambda k: k[-1], reverse = True)
        print(results[0])

        ocsvm = OneClassSVM(**results[0][0])
        ocsvm.fit(insample_train)
        pickle.dump(ocsvm, open(modelname + '_ocsv_model.sav', 'wb'))

    return results[0][2],results[0][3],results

def main(args):

    loadclassifier = args.loadclassifier
    modelname      = args.modelname

    z_mean_insample_train         = np.asarray(loadmat(modelname + '_scores.mat')['z_mean_insample_train']).astype(float)
    z_log_var_insample_train      = np.asarray(loadmat(modelname + '_scores.mat')['z_log_var_insample_train']).astype(float)
    z_mean_insample_test          = np.asarray(loadmat(modelname + '_scores.mat')['z_mean_insample_test']).astype(float)
    z_log_var_insample_test       = np.asarray(loadmat(modelname + '_scores.mat')['z_log_var_insample_test']).astype(float)
    z_mean_outsample_test         = np.asarray(loadmat(modelname + '_scores.mat')['z_mean_outsample_test']).astype(float)
    z_log_var_outsample_test      = np.asarray(loadmat(modelname + '_scores.mat')['z_log_var_outsample_test']).astype(float)

    z_insample_train  = np.hstack((z_mean_insample_train,z_log_var_insample_train))
    z_insample_test   = np.hstack((z_mean_insample_test,z_log_var_insample_test))
    z_outsample_test  = np.hstack((z_mean_outsample_test,z_log_var_outsample_test))

    #######################################################################################################################################################################
    
    print("Using ocsvm (z-mean)")
    z_mean_ocsvm_recall, z_mean_ocsvm_precision, z_mean_ocsvm_results= classify_using_ocsvm((z_mean_insample_train), \
        (z_mean_insample_test,z_mean_outsample_test),loadclassifier,modelname+'_z_mean_')
    print ("oc-svm using z-mean results: recall:({}%) and precision:({}%)".\
        format(z_mean_ocsvm_recall,z_mean_ocsvm_precision))

    print("Using robustcovariance (z-mean)")
    z_mean_robustcovariance_recall, z_mean_robustcovariance_precision, z_mean_robustcovariance_results = classify_using_robustcovariance((z_mean_insample_train), \
        (z_mean_insample_test,z_mean_outsample_test),loadclassifier,modelname+'_z_mean_')
    print ("oc-robustcovariance using z-mean results: recall:({}%) and precision:({}%)".\
        format(z_mean_robustcovariance_recall,z_mean_robustcovariance_precision))

    print("Using isolationforest (z-mean)")
    z_mean_isolationforest_recall, z_mean_isolationforest_precision, z_mean_isolationforest_results = classify_using_isolationforest((z_mean_insample_train), \
        (z_mean_insample_test,z_mean_outsample_test),loadclassifier,modelname+'_z_mean_')
    print ("oc-isolationforest using z-mean results: recall:({}%) and precision:({}%)".\
        format(z_mean_isolationforest_recall,z_mean_isolationforest_precision))

    print("Using localoutlierfactor (z-mean)")
    z_mean_localoutlierfactor_recall, z_mean_localoutlierfactor_precision,z_mean_localoutlierfactor_results = classify_using_localoutlierfactor((z_mean_insample_train), \
        (z_mean_insample_test,z_mean_outsample_test),loadclassifier,modelname+'_z_mean_')
    print ("oc-localoutlierfactor using z-mean results: recall:({}%) and precision:({}%)".\
        format(z_mean_localoutlierfactor_recall,z_mean_localoutlierfactor_precision))

    print("Using ocnn (z-mean)")
    z_mean_ocnn_recall, z_mean_ocnn_precision, z_mean_ocnn_results= classify_using_ocnn((z_mean_insample_train), \
        (z_mean_insample_test,z_mean_outsample_test),loadclassifier,modelname+'_z_mean_')
    print ("oc-nn using z-mean results: recall:({}%) and precision:({}%)".\
        format(z_mean_ocnn_recall,z_mean_ocnn_precision))

    savemat(modelname + '_z_mean_classification_results.mat', {
    'z_mean_ocnn_recall': z_mean_ocnn_recall,
    'z_mean_ocnn_precision': z_mean_ocnn_precision,
    'z_mean_ocnn_results': z_mean_ocnn_results,
    'z_mean_ocsvm_recall': z_mean_ocsvm_recall,
    'z_mean_ocsvm_precision': z_mean_ocsvm_precision,
    'z_mean_ocsvm_results': z_mean_ocsvm_results,
    'z_mean_robustcovariance_recall': z_mean_robustcovariance_recall,
    'z_mean_robustcovariance_precision': z_mean_robustcovariance_precision,
    'z_mean_robustcovariance_results': z_mean_robustcovariance_results,
    'z_mean_isolationforest_recall': z_mean_isolationforest_recall,
    'z_mean_isolationforest_precision': z_mean_isolationforest_precision,
    'z_mean_isolationforest_results': z_mean_isolationforest_results,
    'z_mean_localoutlierfactor_recall': z_mean_localoutlierfactor_recall,
    'z_mean_localoutlierfactor_precision': z_mean_localoutlierfactor_precision,
    'z_mean_localoutlierfactor_results': z_mean_localoutlierfactor_results
    }) 

#######################################################################################################################################################################

    print("Using ocsvm (z)")
    z_ocsvm_recall, z_ocsvm_precision, z_ocsvm_results= classify_using_ocsvm((z_insample_train), \
        (z_insample_test,z_outsample_test),loadclassifier,modelname+'_z_')
    print ("oc-svm using z results: recall:({}%) and precision:({}%)".\
        format(z_ocsvm_recall,z_ocsvm_precision))

    print("Using robustcovariance (z)")
    z_robustcovariance_recall, z_robustcovariance_precision, z_robustcovariance_results = classify_using_robustcovariance((z_insample_train), \
        (z_insample_test,z_outsample_test),loadclassifier,modelname+'_z_')
    print ("oc-robustcovariance using z results: recall:({}%) and precision:({}%)".\
        format(z_robustcovariance_recall,z_robustcovariance_precision))

    print("Using isolationforest (z)")
    z_isolationforest_recall, z_isolationforest_precision, z_isolationforest_results = classify_using_isolationforest((z_insample_train), \
        (z_insample_test,z_outsample_test),loadclassifier,modelname+'_z_')
    print ("oc-isolationforest using z results: recall:({}%) and precision:({}%)".\
        format(z_isolationforest_recall,z_isolationforest_precision))

    print("Using localoutlierfactor (z)")
    z_localoutlierfactor_recall, z_localoutlierfactor_precision,z_localoutlierfactor_results = classify_using_localoutlierfactor((z_insample_train), \
        (z_insample_test,z_outsample_test),loadclassifier,modelname+'_z_')
    print ("oc-localoutlierfactor using z results: recall:({}%) and precision:({}%)".\
        format(z_localoutlierfactor_recall,z_localoutlierfactor_precision))

    print("Using ocnn (z)")
    z_ocnn_recall, z_ocnn_precision, z_ocnn_results= classify_using_ocnn((z_insample_train), \
        (z_insample_test,z_outsample_test),loadclassifier,modelname+'_z_')
    print ("oc-nn using z results: recall:({}%) and precision:({}%)".\
        format(z_ocnn_recall,z_ocnn_precision))

    savemat(modelname + '_z_classification_results.mat', {
    'z_ocnn_recall': z_ocnn_recall,
    'z_ocnn_precision': z_ocnn_precision,
    'z_ocnn_results': z_ocnn_results,
    'z_ocsvm_recall': z_ocsvm_recall,
    'z_ocsvm_precision': z_ocsvm_precision,
    'z_ocsvm_results': z_ocsvm_results,
    'z_robustcovariance_recall': z_robustcovariance_recall,
    'z_robustcovariance_precision': z_robustcovariance_precision,
    'z_robustcovariance_results': z_robustcovariance_results,
    'z_isolationforest_recall': z_isolationforest_recall,
    'z_isolationforest_precision': z_isolationforest_precision,
    'z_isolationforest_results': z_isolationforest_results,
    'z_localoutlierfactor_recall': z_localoutlierfactor_recall,
    'z_localoutlierfactor_precision': z_localoutlierfactor_precision,
    'z_localoutlierfactor_results': z_localoutlierfactor_results
    }) 

def parse_args(argv):   
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--loadclassifier", help="load out-of-distribution calssifier", action='store_true')
    parser.add_argument("--modelname",help="Model Name", default='EBDS_VAEGAN')

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))

