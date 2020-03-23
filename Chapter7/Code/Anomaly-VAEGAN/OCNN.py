# import the necessary packages
import numpy as np
RANDOM_SEED = 1000
np.random.seed(RANDOM_SEED)
import tensorflow as tf
sess = tf.Session()
import keras
from keras import backend as K
K.set_session(sess)
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras import backend as K
from keras.utils import to_categorical
# set the matplotlib backend so figures can be saved in the background
from keras.callbacks import LambdaCallback
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects
from keras.models import load_model

class OC_NN:

    ## Initialise static variables
    INPUT_DIM = 0
    HIDDEN_SIZE = 0
    ACTIVATION_UNIT = "linear"

    def __init__(self, inputdim, hiddenLayerSize, Activation_Unit , modelSavePath,modelname):
        """
        Called when initializing the classifier
        """
        OC_NN.INPUT_DIM   = inputdim
        OC_NN.HIDDEN_SIZE = hiddenLayerSize
        OC_NN.ACTIVATION_UNIT = Activation_Unit
        self.directory = modelSavePath
        self.modelname = modelname 
        self.model = ""
        self.h_size= OC_NN.HIDDEN_SIZE
        global model
        self.r=1.0
        self.kvar=0.0

    @staticmethod
    def custom_ocnn_loss(self,nu, w, V):

        def custom_hinge(y_true, y_pred):

            term1 = 0.5 * tf.reduce_sum(w[0] ** 2)
            term2 = 0.5 * tf.reduce_sum(V[0] ** 2)
            term3 = 1 / nu * K.mean(K.maximum(0.0, self.r - tf.reduce_max(y_pred, axis=1)), axis=-1)
            term4 = -1*self.r
            # yhat assigned to r
            self.r = tf.reduce_max(y_pred, axis=1)
            # r = nuth quantile
            self.r = tf.contrib.distributions.percentile(self.r, q=100 * nu)
            rval = tf.reduce_max(y_pred, axis=1)
            rval = tf.Print(rval, [tf.shape(rval)])

            return (term1 + term2 + term3 + term4)

        return custom_hinge


    @staticmethod
    def build():

        h_size = OC_NN.HIDDEN_SIZE
        Activation_Unit = OC_NN.ACTIVATION_UNIT

        def custom_activation(x):
            return (1 / np.sqrt(h_size)) * tf.cos(x / 0.02)

        get_custom_objects().update({
            'custom_activation':
                Activation(custom_activation)
        })

        # main thread


        model = Sequential()
        ## Define Dense layer from input to hidden
        input_hidden= Dense(h_size, input_dim= OC_NN.INPUT_DIM, kernel_initializer="glorot_normal",name="input_hidden")
        model.add(input_hidden)
        model.add(Activation(custom_activation))

        ## Define Dense layer from hidden  to output
        hidden_ouput = Dense(2,name="hidden_output")
        model.add(hidden_ouput)
        model.add(Activation(Activation_Unit))

        ## Obtain the weights and bias of the layers
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        # w = [w.eval(K.get_session) for w in layer_dict['input_hidden'].weights]
        with sess.as_default():
            w = input_hidden.get_weights()[0]
            bias1 = input_hidden.get_weights()[1]
            V = hidden_ouput.get_weights()[0]
            bias2 = hidden_ouput.get_weights()[1]

        ## Load the pretrained model
        # model = load_model("/Users/raghav/envPython3/experiments/one_class_neural_networks/models/FF_NN/" + "FF_NN_best.h5")
        # return the constructed network architecture
        return [model,w,V,bias1,bias2]
    
    def fit(self,trainX,cont,nEpochs,batch_size):
        # initialize the model
        EPOCHS = nEpochs
        INIT_LR = 1e-8
        BS = batch_size
        nu = cont
        print("[INFO] compiling model...")
        trainY = np.ones(len(trainX))
        trainY = to_categorical(trainY, num_classes=2) ## trainY is not used while training its just used since defining keras custom loss function required it
        [model, w, V, bias1, bias2] = OC_NN.build()
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        ## Obtain weights of layer
        print("[INFO] ",w[0].shape, "input  --> hidden layer weights shape ...")
        print("[INFO] ",V[0].shape, "hidden --> output layer weights shape ...")
        model.compile(loss=OC_NN.custom_ocnn_loss(self,nu, w, V), optimizer=opt,metrics=None)
        output_layers = ['hidden_output']
        model.metrics_tensors += [layer.output for layer in model.layers if layer.name in output_layers]
        # train the network
        print("[INFO] training network...")

        #def printEvaluation(e, logs):
        #    print("evaluation for epoch: " + str(e) )
        #    print("output:",K.print_tensor(model.metrics_tensors[0], message="tensors is: "))


        #callback = LambdaCallback(on_epoch_end=printEvaluation)

        #callbacks = [callback]


        ## Initialize the network with pretrained weights

        H = model.fit(trainX, trainY, shuffle=False, batch_size=BS, epochs=EPOCHS)

        # save the model to disk
        print("[INFO] serializing network and saving trained weights...")
        print("[INFO] Saving model layer weights..." )
        model.save(self.directory+ self.modelname + "_OC_NN.h5")
        with sess.as_default():
            w = model.layers[0].get_weights()[0]
            V = model.layers[2].get_weights()[0]
            np.save(self.directory+"w", w)
            np.save(self.directory +"V", V)

        # print("[INFO] ",type(w) ,w.shape,"type of w...")
        # print("[INFO] ", type(V),V.shape, "type of V...")
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        # plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
        # plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
        plt.title("OC_NN Training Loss and Accuracy on 1's / 7's")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Vs Epochs")
        plt.legend(loc="upper right")
        plt.savefig(self.directory+"trainValLoss.png")
        plt.close()

    def score(self,testNegX,testPosX,nu):
        # load the trained convolutional neural network
        print("[INFO] loading network...")
        w =  np.load(self.directory + "w.npy")
        V =  np.load(self.directory + "V.npy")

        model = load_model(self.directory+ self.modelname + "_OC_NN.h5",custom_objects={'custom_hinge': OC_NN.custom_ocnn_loss(self,nu,w,V)})
        ## Initialize the network with pretrained weights

        y_pred_Pos = model.predict_proba(testPosX)
        y_pred_Pos = np.argmax(y_pred_Pos, axis=1)
        Corr_Pos   = y_pred_Pos[y_pred_Pos==1].size / y_pred_Pos.size

        y_pred_Neg = model.predict_proba(testNegX)
        y_pred_Neg = np.argmax(y_pred_Neg, axis=1)
        Corr_Neg   = y_pred_Neg[y_pred_Neg == 0].size / y_pred_Neg.size

        return Corr_Neg,Corr_Pos
