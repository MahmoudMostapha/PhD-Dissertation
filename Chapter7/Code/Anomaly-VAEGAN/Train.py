from datetime import datetime
import sys
import os
import argparse
from sklearn.model_selection import train_test_split
from scipy.io import loadmat,savemat
from sklearn import preprocessing   
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import layers,models,optimizers,regularizers,losses
import numpy as np
import pandas as pd
from sklearn import metrics
import seaborn as sns
from tensorflow.keras import utils
from tensorflow.keras.utils import get_custom_objects
import matplotlib.pyplot as plt
import matplotlib
print(tf.VERSION)
print(tf.keras.__version__)
from skimage import io
from skimage.transform import resize
import SimpleITK as sitk
from sklearn import metrics
import csv
from My_Classes import DiscriminatorGenerator, DecoderGenerator, EncoderGenerator, OCNNGenerator
from sklearn import manifold
from sklearn.utils import class_weight


# Global Variables
r  = 1.0
seed = 1

def plot_latent(encoder,
                 IDs,
                 Labels,
                 latent_dim,
                 modelname,
                 selectedepoch):

    z_mean =  np.zeros((len(IDs),latent_dim)) 
    Labels = np.asarray(Labels).astype(int)
    idx = 0
    for ID in IDs:
        image       = sitk.ReadImage(ID)
        image_array = np.asarray(sitk.GetArrayFromImage(image)).astype(float)
        Input = (np.expand_dims(image_array, axis=-1) / np.percentile(image_array, 99.5)) - 0.5
        z_mean[idx,:], _, _ = encoder.predict(np.expand_dims(Input, axis=0))

    tsne = manifold.TSNE(n_components=2, init='random',random_state=seed, perplexity=30.0, n_iter=5000)
    z_mean_2D = tsne.fit_transform(z_mean)

    savemat('./' + str(selectedepoch) + '/' + modelname + '_z_mean_2D.mat', {
    'z_mean_2D': z_mean_2D,
    'Labels': Labels
    },do_compression=True) 

    print(z_mean_2D.shape)
    print(Labels.shape)
    z_mean_2D_Insample = z_mean_2D[Labels!=4]
    print(z_mean_2D_Insample.shape)
    Labels_Insample = Labels[Labels!=4]
    print(Labels_Insample.shape)

    fig = plt.figure(figsize=(12, 10))
    plt.scatter(z_mean_2D_Insample[:, 0], z_mean_2D_Insample[:, 1], c = Labels_Insample, cmap='rainbow')
    plt.legend(loc='upper right')
    ax = plt.axes()
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    plt.axis('tight')
    plt.tight_layout()
    #plt.show()
    fig.savefig('./' + str(selectedepoch) + '/' + modelname+'_z_mean_2D_Insample_'+str(selectedepoch)+'_.png', dpi=fig.dpi)
    plt.close()

    Labels[Labels!=4] = 0
    fig = plt.figure(figsize=(12, 10))
    plt.scatter(z_mean_2D[:, 0], z_mean_2D[:, 1], c = Labels, cmap='rainbow')
    plt.legend(loc='upper right')
    ax = plt.axes()
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    plt.axis('tight')
    plt.tight_layout()
    #plt.show()
    fig.savefig('./' + str(selectedepoch) + '/' + modelname+'_z_mean_2D_'+str(selectedepoch)+'_.png', dpi=fig.dpi)
    plt.close()

def Find_Optimal_Cutoff(fpr, tpr, threshold):

    return threshold[np.argmax(tpr + (1 - fpr))],tpr[np.argmax(tpr + (1 - fpr))],1 - fpr[np.argmax(tpr + (1 - fpr))]

def test_scores(scores_insample_train,
                 scores_insample_test,
                 scores_outsample_test,
                 modelname,
                 selectedepoch):

    score_kl_insample_train, score_recons_insample_train, score_ocnn_insample_train    = scores_insample_train 
    score_kl_insample_test, score_recons_insample_test, score_ocnn_insample_test       = scores_insample_test 
    score_kl_outsample_test, score_recons_outsample_test, score_ocnn_outsample_test    = scores_outsample_test  

    # KL Divergence 
    print("Testing using KL Divergence scores (insample pos)")
    y_test_insample  = np.ones((score_kl_insample_test.shape[0],1))
    y_test_outsample = np.zeros((score_kl_outsample_test.shape[0],1))
    y = np.vstack((y_test_insample,y_test_outsample))
    scores = np.vstack((score_kl_insample_test,score_kl_outsample_test))
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    AUC = metrics.auc(fpr, tpr)
    best_T, SEN, SPC = Find_Optimal_Cutoff(fpr, tpr, thresholds)
    print(best_T)
    print(AUC)
    print(SEN)
    print(SPC)
    fig = plt.plot(fpr, tpr, lw=1, alpha=0.3, label="(AUC = {}) \n (Best Threshold = {}) \n (Insample: {}, Outsample: {})".format(AUC, best_T, SEN, SPC))
    plt.legend(loc='upper right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('KL Divergence Score (Insample: Pos)')
    plt.savefig('./' + str(selectedepoch) + '/' + modelname+ "auc_kl_insample_pos.png", dpi=400) 
    plt.close()

    print("Testing using KL Divergence scores (insample Neg)")
    y_test_insample  = np.zeros((score_kl_insample_test.shape[0],1))
    y_test_outsample = np.ones((score_kl_outsample_test.shape[0],1))
    y = np.vstack((y_test_insample,y_test_outsample))
    scores = np.vstack((score_kl_insample_test,score_kl_outsample_test))
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    AUC = metrics.auc(fpr, tpr)
    best_T, SEN, SPC = Find_Optimal_Cutoff(fpr, tpr, thresholds)
    print(best_T)
    print(AUC)
    print(SEN)
    print(SPC)
    fig = plt.plot(fpr, tpr, lw=1, alpha=0.3, label="(AUC = {}) \n (Best Threshold = {}) \n (Insample: {}, Outsample: {})".format(AUC, best_T, SEN, SPC))
    plt.legend(loc='upper right')
    plt.show()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('KL Divergence Score (Insample: Neg)')
    plt.savefig('./' + str(selectedepoch) + '/' + modelname+ "auc_kl_insample_neg.png", dpi=400) 
    plt.close()

    # Reconstruction 
    print("Testing using Reconstruction scores (insample pos)")
    y_test_insample  = np.ones((score_recons_insample_test.shape[0],1))
    y_test_outsample = np.zeros((score_recons_outsample_test.shape[0],1))
    y = np.vstack((y_test_insample,y_test_outsample))
    scores = np.vstack((score_recons_insample_test,score_recons_outsample_test))
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    AUC = metrics.auc(fpr, tpr)
    best_T, SEN, SPC  = Find_Optimal_Cutoff(fpr, tpr, thresholds)
    print(best_T)
    print(AUC)
    print(SEN)
    print(SPC)
    fig = plt.plot(fpr, tpr, lw=1, alpha=0.3, label="(AUC = {}) \n (Best Threshold = {}) \n (Insample: {}, Outsample: {})".format(AUC, best_T, SEN, SPC))
    plt.legend(loc='upper right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Reconstruction Score (Insample: Pos)')
    plt.show() 
    plt.savefig('./' + str(selectedepoch) + '/' + modelname+ "auc_recons_insample_pos.png", dpi=400) 
    plt.close()

    print("Testing using Reconstruction scores (insample Neg)")
    y_test_insample  = np.zeros((score_recons_insample_test.shape[0],1))
    y_test_outsample = np.ones((score_recons_outsample_test.shape[0],1))
    y = np.vstack((y_test_insample,y_test_outsample))
    scores = np.vstack((score_recons_insample_test,score_recons_outsample_test))
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    AUC = metrics.auc(fpr, tpr)
    best_T, SEN, SPC = Find_Optimal_Cutoff(fpr, tpr, thresholds)
    print(best_T)
    print(AUC)
    print(SEN)
    print(SPC)
    fig = plt.plot(fpr, tpr, lw=1, alpha=0.3, label="(AUC = {}) \n (Best Threshold = {}) \n (Insample: {}, Outsample: {})".format(AUC, best_T, SEN, SPC))
    plt.legend(loc='upper right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Reconstruction Score (Insample: Neg)')
    plt.show()  
    plt.savefig('./' + str(selectedepoch) + '/' + modelname+ "auc_recons_insample_neg.png", dpi=400) 
    plt.close()

    # OCNN 
    print("Testing using OCNN scores (insample pos)")
    y_test_insample  = np.ones((score_ocnn_insample_test.shape[0],1))
    y_test_outsample = np.zeros((score_ocnn_outsample_test.shape[0],1))
    y = np.vstack((y_test_insample,y_test_outsample))
    scores = np.vstack((score_ocnn_insample_test,score_ocnn_outsample_test))
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    AUC = metrics.auc(fpr, tpr)
    best_T, SEN, SPC = Find_Optimal_Cutoff(fpr, tpr, thresholds)
    print(best_T)
    print(AUC)
    print(SEN)
    print(SPC)
    fig = plt.plot(fpr, tpr, lw=1, alpha=0.3, label="(AUC = {}) \n (Best Threshold = {}) \n (Insample: {}, Outsample: {})".format(AUC, best_T, SEN, SPC))
    plt.legend(loc='upper right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('OCNN Score (Insample: Pos)')
    plt.show()
    plt.savefig('./' + str(selectedepoch) + '/' + modelname+ "auc_ocnn_insample_pos.png", dpi=400) 
    plt.close()

    print("Testing using OCNN scores (insample Neg)")
    y_test_insample  = np.zeros((score_ocnn_insample_test.shape[0],1))
    y_test_outsample = np.ones((score_ocnn_outsample_test.shape[0],1))
    y = np.vstack((y_test_insample,y_test_outsample))
    scores = np.vstack((score_ocnn_insample_test,score_ocnn_outsample_test))
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    AUC = metrics.auc(fpr, tpr)
    best_T, SEN, SPC = Find_Optimal_Cutoff(fpr, tpr, thresholds)
    print(best_T)
    print(AUC)
    print(SEN)
    print(SPC)
    fig = plt.plot(fpr, tpr, lw=1, alpha=0.3, label="(AUC = {}) \n (Best Threshold = {}) \n (Insample: {}, Outsample: {})".format(AUC, best_T, SEN, SPC))
    plt.legend(loc='upper right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('OCNN Score (Insample: Neg)')
    plt.show() 
    plt.savefig('./' + str(selectedepoch) + '/' + modelname + "auc_ocnn_insample_neg.png", dpi=400) 
    plt.close()

def plot_scores(scores_insample_train,
                 scores_insample_test,
                 scores_outsample_test,
                 modelname,
                 selectedepoch):

    score_kl_insample_train, score_recons_insample_train, score_ocnn_insample_train    = scores_insample_train 
    score_kl_insample_test, score_recons_insample_test, score_ocnn_insample_test       = scores_insample_test 
    score_kl_outsample_test, score_recons_outsample_test, score_ocnn_outsample_test    = scores_outsample_test  

    y_train_insample  = np.zeros((score_kl_insample_train.shape[0],1))
    y_test_insample   = 1 * np.ones((score_kl_insample_test.shape[0],1))
    y_test_outsample  = 2 * np.ones((score_kl_outsample_test.shape[0],1))
    y                 = np.vstack((y_train_insample,y_test_insample,y_test_outsample))

    kl_scores     = np.vstack((score_kl_insample_train,score_kl_insample_test,score_kl_outsample_test))
    recons_scores = np.vstack((score_recons_insample_train,score_recons_insample_test,score_recons_outsample_test))
    ocnn_scores   = np.vstack((score_ocnn_insample_train,score_ocnn_insample_test,score_ocnn_outsample_test))

    Data = np.concatenate((kl_scores,recons_scores,ocnn_scores,y), axis=1)
    df   = pd.DataFrame(data=Data,columns=["score_kl","score_recons","score_ocnn","class"])
    def Renameclass (row):
        if row['class'] == 0 :
            return 'In-sample (training)'
        if row['class'] == 1 :
            return 'In-sample (testing)'
        return 'Out-sample (testing)'

    df['class'] = df.apply (lambda row: Renameclass (row),axis=1)
    df_insample_train  = df.loc[df['class'] == 'In-sample (training)']
    df_insample_test   = df.loc[df['class'] == 'In-sample (testing)']
    df_outsample_test  = df.loc[df['class'] == 'Out-sample (testing)']

    p1_kl=sns.kdeplot(df_insample_train['score_kl'], shade=True)
    p3_kl=sns.kdeplot(df_insample_test['score_kl'], shade=True)
    p5_kl=sns.kdeplot(df_outsample_test['score_kl'], shade=True).set_title("KL Score KDE")
    legend = plt.legend()
    for t, l in zip(legend.texts,("In-sample (training)", "In-sample (testing)","Out-sample (testing)")):
        t.set_text(l)
    plt.show()
    figure1 = p5_kl.get_figure()  
    figure1.savefig('./' + str(selectedepoch) + '/' + modelname + "score_kl.png", dpi=400) 
    plt.close()

    p1_recons=sns.kdeplot(df_insample_train['score_recons'], shade=True)
    p3_recons=sns.kdeplot(df_insample_test['score_recons'], shade=True)
    p5_recons=sns.kdeplot(df_outsample_test['score_recons'], shade=True).set_title("Reconstruction Score KDE")
    legend = plt.legend()
    for t, l in zip(legend.texts,("In-sample (training)", "In-sample (testing)","Out-sample (testing)")):
        t.set_text(l)
    plt.show()
    figure2 = p5_recons.get_figure()  
    figure2.savefig('./' + str(selectedepoch) + '/' + modelname + "score_recons.png", dpi=400) 
    plt.close()

    p1_ocnn=sns.kdeplot(df_insample_train['score_ocnn'], shade=True)
    p3_ocnn=sns.kdeplot(df_insample_test['score_ocnn'], shade=True)
    p5_ocnn=sns.kdeplot(df_outsample_test['score_ocnn'], shade=True).set_title("OCNN Score KDE")
    legend = plt.legend()
    for t, l in zip(legend.texts,("In-sample (training)", "In-sample (testing)","Out-sample (testing)")):
        t.set_text(l)
    plt.show()
    figure3 = p5_ocnn.get_figure()  
    figure3.savefig('./' + str(selectedepoch) + '/' + modelname + "score_ocnn.png", dpi=400) 
    plt.close()

def compute_anomaly_scores(My_models,
                 My_data_train,
                 My_data_test,
                 latent_dim,
                 numberofsamples):

    encoder, decoder, disc, ocnn, classifier = My_models
    Training_Insample_IDs = My_data_train
    Testing_Insample_IDs,Testing_Outsample_IDs = My_data_test

    score_kl_insample_train      =  np.zeros(len(Training_Insample_IDs))
    score_recons_insample_train  =  np.zeros(len(Training_Insample_IDs))
    score_ocnn_insample_train    =  np.zeros(len(Training_Insample_IDs))
    z_mean_insample_train        =  np.zeros((len(Training_Insample_IDs),latent_dim))
    z_log_var_insample_train     =  np.zeros((len(Training_Insample_IDs),latent_dim))   

    idx = 0
    for ID in Training_Insample_IDs:
        image       = sitk.ReadImage(ID)
        image_array = np.asarray(sitk.GetArrayFromImage(image)).astype(float)
        Input = (np.expand_dims(image_array, axis=-1) / np.percentile(image_array, 99.5)) - 0.5
        for i in range(numberofsamples):
            z_mean, z_log_var, z = encoder.predict(np.expand_dims(Input, axis=0))
            score_kl_insample_train[idx] += (-0.5 * np.sum(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=-1))
            Output = decoder.predict(z)
            dis_feat        = disc.predict(np.expand_dims(Input, axis=0))[1]
            dis_feat_tilde  = disc.predict(Output)[1]
            nll = 0.5 * np.log(2 * np.pi) + 0.5 * np.square(dis_feat_tilde - dis_feat)
            axis = tuple(range(1, len(dis_feat_tilde.shape)))
            score_recons_insample_train[idx] += np.sum(nll, axis=axis)
            score_ocnn_insample_train[idx] += ocnn.predict(z_mean).flatten()

        score_kl_insample_train[idx]     = np.divide(score_kl_insample_train[idx], float(numberofsamples))    
        score_recons_insample_train[idx] = np.divide(score_recons_insample_train[idx], float(numberofsamples))   
        score_ocnn_insample_train[idx]   = np.divide(score_ocnn_insample_train[idx], float(numberofsamples)) 
        z_mean_insample_train[idx,:]     =  z_mean
        z_log_var_insample_train[idx,:]  =  z_log_var  
        idx+= 1 

    score_kl_insample_test      =  np.zeros(len(Testing_Insample_IDs))
    score_recons_insample_test  =  np.zeros(len(Testing_Insample_IDs))
    score_ocnn_insample_test    =  np.zeros(len(Testing_Insample_IDs))
    z_mean_insample_test        =  np.zeros((len(Testing_Insample_IDs),latent_dim))
    z_log_var_insample_test     =  np.zeros((len(Testing_Insample_IDs),latent_dim))   

    idx = 0
    for ID in Testing_Insample_IDs:
        image       = sitk.ReadImage(ID)
        image_array = np.asarray(sitk.GetArrayFromImage(image)).astype(float)
        Input = (np.expand_dims(image_array, axis=-1) / np.percentile(image_array, 99.5)) - 0.5
        for i in range(numberofsamples):
            z_mean, z_log_var, z = encoder.predict(np.expand_dims(Input, axis=0))
            score_kl_insample_test[idx] += (-0.5 * np.sum(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=-1))
            Output = decoder.predict(z)
            dis_feat        = disc.predict(np.expand_dims(Input, axis=0))[1]
            dis_feat_tilde  = disc.predict(Output)[1]
            nll = 0.5 * np.log(2 * np.pi) + 0.5 * np.square(dis_feat_tilde - dis_feat)
            axis = tuple(range(1, len(dis_feat_tilde.shape)))
            score_recons_insample_test[idx] += np.sum(nll, axis=axis)
            score_ocnn_insample_test[idx] += ocnn.predict(z_mean).flatten()

        score_kl_insample_test[idx]     = np.divide(score_kl_insample_test[idx], float(numberofsamples))    
        score_recons_insample_test[idx] = np.divide(score_recons_insample_test[idx], float(numberofsamples))   
        score_ocnn_insample_test[idx]   = np.divide(score_ocnn_insample_test[idx], float(numberofsamples)) 
        z_mean_insample_test[idx,:]     =  z_mean
        z_log_var_insample_test[idx,:]  =  z_log_var 
        idx+= 1 

    score_kl_outsample_test      =  np.zeros(len(Testing_Outsample_IDs))
    score_recons_outsample_test  =  np.zeros(len(Testing_Outsample_IDs))
    score_ocnn_outsample_test    =  np.zeros(len(Testing_Outsample_IDs))
    z_mean_outsample_test        =  np.zeros((len(Testing_Outsample_IDs),latent_dim))
    z_log_var_outsample_test     =  np.zeros((len(Testing_Outsample_IDs),latent_dim))   

    idx = 0
    for ID in Testing_Outsample_IDs:
        image       = sitk.ReadImage(ID)
        image_array = np.asarray(sitk.GetArrayFromImage(image)).astype(float)
        Input = (np.expand_dims(image_array, axis=-1) / np.percentile(image_array, 99.5)) - 0.5
        for i in range(numberofsamples):
            z_mean, z_log_var, z = encoder.predict(np.expand_dims(Input, axis=0))
            score_kl_outsample_test[idx] += (-0.5 * np.sum(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=-1))
            Output = decoder.predict(z)
            dis_feat        = disc.predict(np.expand_dims(Input, axis=0))[1]
            dis_feat_tilde  = disc.predict(Output)[1]
            nll = 0.5 * np.log(2 * np.pi) + 0.5 * np.square(dis_feat_tilde - dis_feat)
            axis = tuple(range(1, len(dis_feat_tilde.shape)))
            score_recons_outsample_test[idx] += np.sum(nll, axis=axis)
            score_ocnn_outsample_test[idx] += ocnn.predict(z_mean).flatten()

        score_kl_outsample_test[idx]     = np.divide(score_kl_outsample_test[idx], float(numberofsamples))    
        score_recons_outsample_test[idx] = np.divide(score_recons_outsample_test[idx], float(numberofsamples))   
        score_ocnn_outsample_test[idx]   = np.divide(score_ocnn_outsample_test[idx], float(numberofsamples)) 
        z_mean_outsample_test[idx,:]     =  z_mean
        z_log_var_outsample_test[idx,:]  =  z_log_var
        idx+= 1   

    scores_insample_train  = (score_kl_insample_train.reshape(-1,1), score_recons_insample_train.reshape(-1,1),score_ocnn_insample_train.reshape(-1,1))
    scores_insample_test   = (score_kl_insample_test.reshape(-1,1), score_recons_insample_test.reshape(-1,1),score_ocnn_insample_test.reshape(-1,1))
    scores_outsample_test  = (score_kl_outsample_test.reshape(-1,1), score_recons_outsample_test.reshape(-1,1),score_ocnn_outsample_test.reshape(-1,1))
    latent_insample_train  = (z_mean_insample_train, z_log_var_insample_train)
    latent_insample_test   = (z_mean_insample_test, z_log_var_insample_test)
    latent_outsample_test  = (z_mean_outsample_test, z_log_var_outsample_test)

    return scores_insample_train, scores_insample_test, scores_outsample_test, latent_insample_train, latent_insample_test, latent_outsample_test

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def mean_gaussian_negative_log_likelihood(y, y_pred):
    nll = 0.5 * np.log(2 * np.pi) + 0.5 * np.square(y_pred - y)
    axis = tuple(range(1, len(tf.keras.backend.int_shape(y))))
    return tf.reduce_mean(tf.reduce_sum(nll, axis=axis), axis=-1)

def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def custom_ocnn_loss(nu, w, V):

    def custom_hinge(y, y_pred):

        global r

        term1 = 0.5 * tf.reduce_sum(w[0] ** 2)
        term2 = 0.5 * tf.reduce_sum(V[0] ** 2)
        term3 = 1 / nu * tf.reduce_mean(tf.maximum(0.0, r - tf.reduce_max(y_pred, axis=1)), axis=-1)
        term4 = -1*r
        # yhat assigned to r
        r = tf.reduce_max(y_pred, axis=1)
        # r = nuth quantile
        r = tf.contrib.distributions.percentile(r, q=100 * nu)

        return (term1 + term2 + term3 + term4)

    return custom_hinge


def main(args):

    usedropout             = args.usedropout
    dropoutrate            = args.dropoutrate
    batch_size             = args.batch_size
    latent_dim             = args.latent_dim
    epochs                 = args.epochs
    steps_epoch            = args.steps_epoch
    encoder_epochs         = args.encoder_epochs
    decoder_epochs         = args.decoder_epochs
    discriminator_epochs   = args.discriminator_epochs
    vae_epochs             = args.vae_epochs
    ocnn_epochs            = args.ocnn_epochs
    recon_vs_gan_weight    = args.recon_vs_gan_weight
    classifier_weight      = args.classifier_weight
    numberofsamples        = args.numberofsamples
    loadmodel              = args.loadmodel
    loadscores             = args.loadscores
    plotscores             = args.plotscores
    plotlatent             = args.plotlatent
    testscores             = args.testscores
    modelname              = args.modelname
    selectedepoch          = args.selectedepoch

    # build encoder model

    def encoder(kernel, filter, rows, columns, depth, channel, latent_dim, usedropout=False, dropoutrate=0.2):

        inputs_x = layers.Input(shape=(rows, columns, depth, channel), name='data_input')
        model    = layers.Conv3D(filters=filter, kernel_size=kernel, strides=(2,2,2), padding='same',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(inputs_x)
        model    = layers.BatchNormalization(epsilon=1e-6)(model)
        model    = layers.LeakyReLU(0.2)(model)
        if usedropout:
            model    = layers.Dropout(dropoutrate)(model)
        model    = layers.Conv3D(filters=2*filter, kernel_size=kernel, strides=(2,2,2), padding='same', kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        model    = layers.BatchNormalization(epsilon=1e-6)(model)
        model    = layers.LeakyReLU(0.2)(model)
        if usedropout:
            model    = layers.Dropout(dropoutrate)(model)
        model    = layers.Conv3D(filters=4*filter, kernel_size=kernel, strides=(2,2,2), padding='same', kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        model    = layers.BatchNormalization(epsilon=1e-6)(model)
        model    = layers.LeakyReLU(0.2)(model)
        if usedropout:
            model    = layers.Dropout(dropoutrate)(model)
        model    = layers.Conv3D(filters=8*filter, kernel_size=kernel, strides=(2,2,2), padding='same',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        model    = layers.BatchNormalization(epsilon=1e-6)(model)
        model    = layers.LeakyReLU(0.2)(model)
        if usedropout:
            model    = layers.Dropout(dropoutrate)(model)
        model    = layers.Conv3D(filters=16*filter, kernel_size=kernel, strides=(2,2,2), padding='same',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        model    = layers.BatchNormalization(epsilon=1e-6)(model)
        model    = layers.LeakyReLU(0.2)(model)

        model     = layers.Flatten()(model)
        z_mean    = layers.Dense(latent_dim,name='z_mean',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        z_log_var = layers.Dense(latent_dim,name='z_log_var',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)

        # use reparameterization trick to push the sampling out as input
        z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = models.Model(inputs_x, [z_mean, z_log_var, z], name='encoder')
        return encoder

    # build decoder model
    def decoder(kernel, filter, z_rows, z_columns, z_depth, z_channel, x_channel, latent_dim, usedropout=False,dropoutrate=0.2):

        inputs_z = layers.Input(shape=(latent_dim,), name='z_sampling')
        model    = layers.Dense(z_rows*z_columns*z_depth*z_channel , kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(inputs_z)
        model    = layers.Reshape((z_rows,z_columns,z_depth,z_channel))(model)
        model    = layers.BatchNormalization(epsilon=1e-6)(model)
        model    = layers.LeakyReLU(0.2)(model)
        if usedropout:
            model    = layers.Dropout(dropoutrate)(model)
        model    = layers.Conv3DTranspose(filters=filter*16, kernel_size=kernel, strides=(2,2,2), padding='same',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        model    = layers.BatchNormalization(epsilon=1e-6)(model)
        model    = layers.LeakyReLU(0.2)(model)
        if usedropout:
            model    = layers.Dropout(dropoutrate)(model)
        model    = layers.Conv3DTranspose(filters=filter*8, kernel_size=kernel, strides=(2,2,2), padding='same',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        model    = layers.BatchNormalization(epsilon=1e-6)(model)
        model    = layers.LeakyReLU(0.2)(model)
        if usedropout:
            model    = layers.Dropout(dropoutrate)(model)
        model    = layers.Conv3DTranspose(filters=filter*4, kernel_size=kernel, strides=(2,2,2), padding='same',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        model    = layers.BatchNormalization(epsilon=1e-6)(model)
        model    = layers.LeakyReLU(0.2)(model)
        if usedropout:
            model    = layers.Dropout(dropoutrate)(model)
        model    = layers.Conv3DTranspose(filters=filter*2, kernel_size=kernel, strides=(2,2,2), padding='same',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        model    = layers.BatchNormalization(epsilon=1e-6)(model)
        model    = layers.LeakyReLU(0.2)(model)
        if usedropout:
            model    = layers.Dropout(dropoutrate)(model)
        model    = layers.Conv3DTranspose(filters=x_channel, kernel_size=kernel, strides=(2,2,2), padding='same',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        #model    = layers.BatchNormalization(epsilon=1e-6)(model)
        out      = layers.Activation('tanh', name='recons_input')(model)    

        # instantiate decoder model
        decoder = models.Model(inputs_z, out, name='decoder')
        return decoder

    def discriminator(kernel, filter, rows, columns, depth, channel, usedropout=False, dropoutrate=0.2):
        
        inputs_x = layers.Input(shape=(rows, columns, depth, channel), name='data_input')
        model    = layers.Conv3D(filters=filter, kernel_size=kernel, strides=(2,2,2), padding='same',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(inputs_x)
        #model    = layers.BatchNormalization(epsilon=1e-6)(model)
        model    = layers.LeakyReLU(0.2)(model)
        if usedropout:
            model    = layers.Dropout(dropoutrate)(model)
        model    = layers.Conv3D(filters=2*filter, kernel_size=kernel, strides=(2,2,2), padding='same', kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        model    = layers.BatchNormalization(epsilon=1e-6)(model)
        model    = layers.LeakyReLU(0.2)(model)
        if usedropout:
            model    = layers.Dropout(dropoutrate)(model)
        model    = layers.Conv3D(filters=4*filter, kernel_size=kernel, strides=(2,2,2), padding='same',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        model    = layers.BatchNormalization(epsilon=1e-6)(model)
        model    = layers.LeakyReLU(0.2)(model)
        if usedropout:
            model     = layers.Dropout(dropoutrate)(model)
        model     = layers.Conv3D(filters=8*filter, kernel_size=kernel, strides=(2,2,2), padding='same',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        model     = layers.BatchNormalization(epsilon=1e-6)(model)
        model     = layers.LeakyReLU(0.2)(model)
        if usedropout:
            model     = layers.Dropout(dropoutrate)(model)
        model     = layers.Conv3D(filters=16*filter, kernel_size=kernel, strides=(2,2,2), padding='same',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(model)
        model     = layers.BatchNormalization(epsilon=1e-6)(model)
        model_out = layers.LeakyReLU(0.2)(model)

        if usedropout:
            disc  = layers.Dropout(dropoutrate)(model_out)
        disc      = layers.Flatten()(disc)
        out       = layers.Dense(5, activation='softmax',kernel_regularizer=regularizers.l2(1e-5), kernel_initializer='he_uniform')(disc)

        # instantiate discriminator models
        discriminator = models.Model(inputs_x, [out,model_out] , name='discriminator')
        return discriminator

    def classifier(inputdim, hiddenLayerSize):
        
        inputs_z_mean = layers.Input(shape=(inputdim,) , name='z_mean')
        model = layers.Dense(hiddenLayerSize, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-5))(inputs_z_mean)
        out   = layers.Dense(4, activation='softmax', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-5), name='label_probs')(model)
        
        # instantiate classifier models
        classifier = models.Model(inputs_z_mean, out, name='classifier')
        return classifier

    def ocnn(inputdim, hiddenLayerSize):

        #def custom_activation(x):
        #    return (1 / np.sqrt(hiddenLayerSize)) * tf.cos(x / 0.02)
        #
        #get_custom_objects().update({
        #   'custom_activation':
        #        layers.Activation(custom_activation)
        #})
        
        inputs_z = layers.Input(shape=(inputdim, ), name='z_input')
        input_hidden = layers.Dense(hiddenLayerSize, kernel_initializer='he_uniform')
        input_hidden_out  = input_hidden(inputs_z)
        input_hidden_out  = layers.Activation('linear', name='input_hidden')(input_hidden_out)
        score     = layers.Dense(1, kernel_initializer='he_uniform')
        score_out = score(input_hidden_out)
        out = layers.Activation('linear', name='output_score')(score_out)

        w = input_hidden.get_weights()[0]
        bias1 = input_hidden.get_weights()[1]
        V = score.get_weights()[0]
        bias2 = score.get_weights()[1]

        # instantiate ocnn model
        ocnn = models.Model(inputs_z, out , name='ocnn')

        return ocnn, w, V, bias1, bias2

    # Model Hyperparameters
    W = 128
    H = 128
    D = 128
    C = 1
    ADAMop = optimizers.Adam(lr = 0.0001)
    ADAMop_OCNN = optimizers.Adam(lr = 0.00001)

    # instantiate encoder model
    Enc = encoder((3,3,3), 32, W, H, D, C, latent_dim, usedropout, dropoutrate)
    Enc.summary()

    # instantiate decoder model
    Dec = decoder((3,3,3), 32, 4, 4, 4, 512, C, latent_dim, usedropout,dropoutrate)
    Dec.summary()

    # instantiate discriminator model
    Disc = discriminator((3,3,3), 32, W, H, D, C,True,dropoutrate)
    Disc.summary()

    # instantiate classifier model
    Class = classifier(latent_dim,128)
    Class.summary()

    # instantiate ocnn model
    OCNN, w, V, bias1, bias2 = ocnn(latent_dim,128)
    OCNN.summary()

    # Build Graph
    inputs     = layers.Input(shape=(W, H, D, C))
    z_mean, z_log_var, z = Enc(inputs)
    outputs = Dec(z)
    Out_real, dis_feat = Disc(inputs)
    Out_fake, dis_feat_tilde = Disc(outputs) 
    z_gen = layers.Input(tensor=tf.random_normal(shape=tf.shape(z)))
    noise_img = Dec(z_gen)
    Out_noise = Disc(noise_img)[0]
    Out_class = Class(z_mean)
    Out_score = OCNN(z_mean)

    # Compute losses

    # Learned similarity metric
    dis_nll_loss = mean_gaussian_negative_log_likelihood(dis_feat, dis_feat_tilde)

    # KL divergence loss
    kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_var - np.square(z_mean) - tf.exp(z_log_var), axis=-1))

    # Create models for training
    encoder_train = models.Model(inputs, [dis_feat_tilde , Out_class], name='Encoder_Train')
    encoder_train.add_loss(tf.reduce_mean(kl_loss + dis_nll_loss))

    decoder_train = models.Model([inputs, z_gen], [Out_fake, Out_noise], name='Decoder_Train')
    normalized_weight = recon_vs_gan_weight / (1. - recon_vs_gan_weight)
    decoder_train.add_loss(normalized_weight * dis_nll_loss)

    discriminator_train = models.Model([inputs, z_gen], [Out_real, Out_fake, Out_noise], name='Discriminator_Train')

    ocnn_train = models.Model(inputs , Out_score ,name='OCNN_Train')

    # Additional models for testing
    vae    = models.Model(inputs, outputs, name='VAE')
    vaegan = models.Model(inputs, Out_fake, name='VAEGAN')

    set_trainable(Enc,False)
    set_trainable(Dec,False)
    set_trainable(Class,False)
    set_trainable(OCNN,False)
    discriminator_train.compile(ADAMop, ['categorical_crossentropy'] * 3,)
    discriminator_train.summary()

    set_trainable(Disc,False)
    set_trainable(Dec,True)
    decoder_train.compile(ADAMop, ['categorical_crossentropy'] * 2)
    decoder_train.summary()

    set_trainable(Enc,True)
    set_trainable(Class,True)
    set_trainable(Dec,False)
    encoder_train.compile(ADAMop, [None,'categorical_crossentropy'], loss_weights = [None,classifier_weight])
    encoder_train.summary()

    set_trainable(OCNN,True)
    ocnn_train.compile(ADAMop_OCNN,loss=custom_ocnn_loss(0.15, w, V))
    ocnn_train.summary()

    set_trainable(vaegan,True)

    # fit models 

    if loadmodel:

        encoder_train.load_weights(modelname+'_encoder_train_weights{0:04d}'.format(selectedepoch)+'.h5')
        decoder_train.load_weights(modelname+'_decoder_train_weights{0:04d}'.format(selectedepoch)+'.h5')
        discriminator_train.load_weights(modelname+'_discriminator_train_weights{0:04d}'.format(selectedepoch)+'.h5')
        ocnn_train.load_weights(modelname+'_ocnn_train_weights{0:04d}'.format(selectedepoch)+'.h5')
        vae.load_weights(modelname+'_vae_weights{0:04d}'.format(selectedepoch)+'.h5')
        vaegan.load_weights(modelname+'_vaegan_weights{0:04d}'.format(selectedepoch)+'.h5')

        Training_IDs             = [line.rstrip('\n') for line in open('./Training_IDs.txt')]
        Training_Insample_IDs    = [line.rstrip('\n') for line in open('./Training_Insample_IDs.txt')]
        Testing_Insample_IDs     = [line.rstrip('\n') for line in open('./Testing_Insample_IDs.txt')]
        Testing_Outsample_IDs    = [line.rstrip('\n') for line in open('./Testing_Outsample_IDs.txt')]

    else:


        Training_IDs             = [line.rstrip('\n') for line in open('./Training_IDs.txt')]
        Training_Insample_IDs    = [line.rstrip('\n') for line in open('./Training_Insample_IDs.txt')]
        Testing_Insample_IDs     = [line.rstrip('\n') for line in open('./Testing_Insample_IDs.txt')]
        Testing_Outsample_IDs    = [line.rstrip('\n') for line in open('./Testing_Outsample_IDs.txt')]

        labels = {}
        reader = csv.reader(open("Training_Labels.csv", "r"))
        for rows in reader:
            k = rows[0]
            v = int(rows[1])
            labels[k] = v

        disc_loss  = []
        dec_loss   = []
        enc_loss   = []
        ocnn_loss  = []

        steps = np.floor(len(Training_IDs) / float(batch_size)).astype(int)

        for epoch in range(epochs):

            np.random.shuffle(Training_IDs)

            print("epoch number {}".format(epoch))

            for step in range(steps):

                Training_IDs_step = Training_IDs[step*batch_size:(step+1)*batch_size]
                print(len(Training_IDs_step))

                # Generators Parameters
                training_params = {'dim': (W,H,D),
                  'batch_size': batch_size,
                  'n_channels': C,
                  'shuffle': False}

                smooth = 0.1

                # Generators
                discriminator_generator = DiscriminatorGenerator(Training_IDs_step, labels, smooth, **training_params)
                decoder_generator = DecoderGenerator(Training_IDs_step, labels, smooth, **training_params)
                encoder_generator = EncoderGenerator(Training_IDs_step, labels, **training_params)

                print("Training Discriminator {} {}".format(epoch,step))

                # train the discriminator
                history_disc = discriminator_train.fit_generator(generator=discriminator_generator,
                        epochs=discriminator_epochs,
                        use_multiprocessing=False,
                        workers=1)

                for i in range(vae_epochs):

                    print("Training Decoder {} {} {}".format(epoch,step,i))

                    # train the decoder
                    history_dec = decoder_train.fit_generator(generator=decoder_generator,
                        epochs=decoder_epochs,
                        use_multiprocessing=False,
                        workers=1)

                    print("Training Encoder {} {} {}".format(epoch,step,i))

                    # train the encoder
                    history_enc = encoder_train.fit_generator(generator=encoder_generator,
                        epochs=encoder_epochs,
                        use_multiprocessing=False,
                        workers=1)

            disc_loss = np.append(disc_loss,history_disc.history['loss'])
            dec_loss = np.append(dec_loss,history_dec.history['loss'])
            enc_loss = np.append(enc_loss,history_enc.history['loss'])

            if (epoch % (epochs - 1)) == 0 and epoch > 0:

                print("Training OCNN {}".format(epoch))

                ocnn_generator    = OCNNGenerator(Training_IDs, **training_params)

                # train the ocnn
                history_ocnn = ocnn_train.fit_generator(generator=ocnn_generator,
                    epochs=ocnn_epochs,
                    use_multiprocessing=False,
                    workers=1)

                ocnn_loss = np.append(ocnn_loss,history_ocnn.history['loss'])
                ocnn_train.save_weights(modelname+'_ocnn_train_weights{0:04d}'.format(epoch)+'.h5')
                ocnn_loss = np.asarray(ocnn_loss)
                savemat(modelname + '_ocnn_loss_curves_{0:04d}.mat'.format(epoch), {'ocnn_loss': ocnn_loss}) 

                fig = plt.figure()
                plt.plot(ocnn_loss)
                plt.title('OCNN Loss Curves')  
                plt.ylabel('OCNN Loss')
                plt.xlabel('Epoch')
                plt.legend(['training set'], loc='upper right')
                plt.tight_layout()
                plt.show()
                fig.savefig(modelname+'_OCNN_history_'+str(epoch)+'_.png', dpi=fig.dpi)
                plt.close()

            if (epoch % 25) == 0 and epoch > 0:
                # Save Models 
                encoder_train.save_weights(modelname+'_encoder_train_weights{0:04d}'.format(epoch)+'.h5')
                decoder_train.save_weights(modelname+'_decoder_train_weights{0:04d}'.format(epoch)+'.h5')
                discriminator_train.save_weights(modelname+'_discriminator_train_weights{0:04d}'.format(epoch)+'.h5')
                ocnn_train.save_weights(modelname+'_ocnn_train_weights{0:04d}'.format(epoch)+'.h5')
                vae.save_weights(modelname+'_vae_weights{0:04d}'.format(epoch)+'.h5')
                vaegan.save_weights(modelname+'_vaegan_weights{0:04d}'.format(epoch)+'.h5')

                disc_loss       = np.asarray(disc_loss)
                dec_loss        = np.asarray(dec_loss)
                enc_loss        = np.asarray(enc_loss)
                savemat(modelname + '_loss_curves_{0:04d}.mat'.format(epoch), {
                'disc_loss': disc_loss,
                'dec_loss': dec_loss,
                'enc_loss': enc_loss
                }) 

                fig = plt.figure()
                plt.plot(disc_loss)
                plt.title('Discriminator Loss Curves')  
                plt.ylabel('Discriminator Loss')
                plt.xlabel('Epoch')
                plt.legend(['training set'], loc='upper right')
                plt.tight_layout()
                plt.show()
                fig.savefig(modelname+'_Discriminator_history_'+str(epoch)+'_.png', dpi=fig.dpi)
                plt.close()

                fig = plt.figure()
                plt.plot(dec_loss[::vae_epochs])
                plt.title('Decoder Loss Curves')  
                plt.ylabel('Decoder Loss')
                plt.xlabel('Epoch')
                plt.legend(['training set'], loc='upper right')
                plt.tight_layout()
                plt.show()
                fig.savefig(modelname+'_Decoder_history_'+str(epoch)+'_.png', dpi=fig.dpi)
                plt.close()

                fig = plt.figure()
                plt.plot(enc_loss[::vae_epochs])
                plt.title('Encoder Loss Curves')  
                plt.ylabel('Encoder Loss')
                plt.xlabel('Epoch')
                plt.legend(['training set'], loc='upper right')
                plt.tight_layout()
                plt.show()
                fig.savefig(modelname+'_Encoder_history_'+str(epoch)+'_.png', dpi=fig.dpi)
                plt.close()

                fig = plt.figure()
                plt.plot(disc_loss)
                plt.plot(dec_loss[::vae_epochs])
                plt.plot(enc_loss[::vae_epochs])
                plt.title('VAEGAN Loss Curves')  
                plt.ylabel('VAEGAN Loss')
                plt.xlabel('Epoch')
                plt.legend(['Discriminator', 'Decoder', 'Encoder'], loc='upper right')
                plt.tight_layout()
                plt.show()
                fig.savefig(modelname+'_VAEGAN_history_'+str(epoch)+'_.png', dpi=fig.dpi)
                plt.close()


    My_models     = (Enc, Dec, Disc, OCNN, Class)
    My_data_train = (Training_Insample_IDs)
    My_data_test  = (Testing_Insample_IDs,Testing_Outsample_IDs)

    if not os.path.exists(str(selectedepoch)):
        os.makedirs(str(selectedepoch))

    if loadscores:

        score_kl_insample_train      = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['score_kl_insample_train']).astype(float)
        score_recons_insample_train  = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['score_recons_insample_train']).astype(float)
        score_ocnn_insample_train    = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['score_ocnn_insample_train']).astype(float)
        score_kl_insample_test       = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['score_kl_insample_test']).astype(float)
        score_recons_insample_test   = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['score_recons_insample_test']).astype(float)
        score_ocnn_insample_test     = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['score_ocnn_insample_test']).astype(float)
        score_kl_outsample_test      = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['score_kl_outsample_test']).astype(float)
        score_recons_outsample_test  = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['score_recons_outsample_test']).astype(float)
        score_ocnn_outsample_test    = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['score_ocnn_outsample_test']).astype(float)

        z_mean_insample_train        = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['z_mean_insample_train']).astype(float)
        z_log_var_insample_train     = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['z_log_var_insample_train']).astype(float)
        z_mean_insample_test         = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['z_mean_insample_test']).astype(float)
        z_log_var_insample_test      = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['z_log_var_insample_test']).astype(float)
        z_mean_outsample_test        = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['z_mean_outsample_test']).astype(float)
        z_log_var_outsample_test     = np.asarray(loadmat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat')['z_log_var_outsample_test']).astype(float)

        scores_insample_train  = (score_kl_insample_train, score_recons_insample_train, score_ocnn_insample_train) 
        scores_insample_test   = (score_kl_insample_test, score_recons_insample_test, score_ocnn_insample_test) 
        scores_outsample_test  = (score_kl_outsample_test, score_recons_outsample_test, score_ocnn_outsample_test) 

        latent_insample_train  = (z_mean_insample_train, z_log_var_insample_train)
        latent_insample_test   = (z_mean_insample_test, z_log_var_insample_test)
        latent_outsample_test  = (z_mean_outsample_test, z_log_var_outsample_test)

    else:

        scores_insample_train, scores_insample_test, scores_outsample_test, \
        latent_insample_train, latent_insample_test, latent_outsample_test \
        = compute_anomaly_scores(My_models, My_data_train, My_data_test, latent_dim, numberofsamples)

        score_kl_insample_train, score_recons_insample_train, score_ocnn_insample_train = scores_insample_train  
        score_kl_insample_test, score_recons_insample_test, score_ocnn_insample_test = scores_insample_test  
        score_kl_outsample_test, score_recons_outsample_test, score_ocnn_outsample_test = scores_outsample_test  

        z_mean_insample_train, z_log_var_insample_train   = latent_insample_train 
        z_mean_insample_test, z_log_var_insample_test     = latent_insample_test   
        z_mean_outsample_test, z_log_var_outsample_test   = latent_outsample_test  

        savemat('./' + str(selectedepoch) + '/' + modelname + '_scores.mat', {
            'score_kl_insample_train': score_kl_insample_train,
            'score_recons_insample_train': score_recons_insample_train,
            'score_ocnn_insample_train': score_ocnn_insample_train,
            'score_kl_insample_test': score_kl_insample_test,
            'score_recons_insample_test': score_recons_insample_test,
            'score_ocnn_insample_test': score_ocnn_insample_test,
            'score_kl_outsample_test': score_kl_outsample_test,
            'score_recons_outsample_test': score_recons_outsample_test,
            'score_ocnn_outsample_test': score_ocnn_outsample_test,
            'z_mean_insample_train': z_mean_insample_train,
            'z_log_var_insample_train': z_log_var_insample_train,
            'z_mean_insample_test': z_mean_insample_test,
            'z_log_var_insample_test': z_log_var_insample_test,
            'z_mean_outsample_test': z_mean_outsample_test,
            'z_log_var_outsample_test': z_log_var_outsample_test
            },do_compression=True) 

    if plotscores:
        plot_scores(scores_insample_train, scores_insample_test, scores_outsample_test, modelname, selectedepoch)

    if testscores:
        test_scores(scores_insample_train, scores_insample_test, scores_outsample_test, modelname, selectedepoch)

    if plotlatent:
        IDs  = [line.rstrip('\n') for line in open('./plot_latent_IDs.txt')]
        Labels = [line.rstrip('\n') for line in open('./plot_latent_Labels.txt')]
        plot_latent(Enc, IDs, Labels, latent_dim, modelname, selectedepoch)


def parse_args(argv):   
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, help="number of epoch", default=1002)
    parser.add_argument("--steps_epoch", type=int, help="number of batches per epoch", default=1)
    parser.add_argument("--encoder_epochs", type=int, help="discriminator number of epoch", default=1)
    parser.add_argument("--decoder_epochs", type=int, help="decoder number of epoch", default=1)
    parser.add_argument("--discriminator_epochs", type=int, help="discriminator number of epoch", default=1)
    parser.add_argument("--vae_epochs", type=int, help="vae number of epoch", default=1)
    parser.add_argument("--ocnn_epochs", type=int, help="ocnn number of epoch", default=100)
    parser.add_argument("--usedropout", help="Use dropout for regulization", action='store_true')
    parser.add_argument("--dropoutrate", help="dropout rate", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, help="mini batch size", default=8)
    parser.add_argument("--latent_dim", type=int, help="latent represenatation dimension", default=300)
    parser.add_argument("--recon_vs_gan_weight", type=float, help="Reconstruction loss weight",default=1e-4)
    parser.add_argument("--classifier_weight", type=float, help="Classifier loss weight",default=100.0)
    parser.add_argument("--numberofsamples",type=int,help="Number of samples used for anomaly scores", default=20)
    parser.add_argument("--loadmodel", help="Load h5 model trained weights",action='store_true')
    parser.add_argument("--selectedepoch",type=int,help="model epoch number to load", default=1000)
    parser.add_argument("--loadscores", help="Load trained model scores",action='store_true')   
    parser.add_argument("--plotscores", help="Plot likihood Scores",action='store_true')
    parser.add_argument("--testscores", help="Test scores for classification",action='store_true')
    parser.add_argument("--plotlatent", help="Plot 2D tsne of latent code",action='store_true')
    parser.add_argument("--modelname",help="Model Name", default='EBDS_VAEGAN')

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))






