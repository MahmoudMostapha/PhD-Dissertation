import numpy as np
import tensorflow as tf
import SimpleITK as sitk

class DiscriminatorGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, smooth, batch_size=8, dim=(128,128,128), n_channels=1, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.labels = labels
        self.smooth = smooth
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, [y_real, y_fake_1 , y_fake_2] = self.__data_generation(list_IDs_temp)

        return X, [y_real, y_fake_1 , y_fake_2]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y_real = np.empty((self.batch_size,), dtype=int)
        y_fake_1 = np.zeros((self.batch_size,), dtype=int)
        y_fake_2 = np.zeros((self.batch_size,), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store sample
            image       = sitk.ReadImage(ID)
            image_array = np.asarray(sitk.GetArrayFromImage(image)).astype(float)
            X[i,] = (np.expand_dims(image_array, axis=-1) / np.percentile(image_array, 99.5)) - 0.5

            # Store class
            y_real[i] = self.labels[ID] + 1

        return X, [tf.keras.utils.to_categorical(y_real, num_classes=5)*(1-self.smooth), tf.keras.utils.to_categorical(y_fake_1, num_classes=5) , tf.keras.utils.to_categorical(y_fake_2, num_classes=5)]

class DecoderGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, smooth, batch_size=8, dim=(128,128,128), n_channels=1, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.labels = labels
        self.smooth = smooth
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, [y_real_1, y_real_2] = self.__data_generation(list_IDs_temp)

        return X, [y_real_1, y_real_2]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y_real_1 = np.empty((self.batch_size,), dtype=int)
        y_real_2 = np.empty((self.batch_size,), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store sample
            image       = sitk.ReadImage(ID)
            image_array = np.asarray(sitk.GetArrayFromImage(image)).astype(float)
            X[i,] = (np.expand_dims(image_array, axis=-1) / np.percentile(image_array, 99.5)) - 0.5

            # Store class
            y_real_1[i] = self.labels[ID] + 1
            y_real_2[i] = self.labels[ID] + 1

        return X, [tf.keras.utils.to_categorical(y_real_1, num_classes=5) * (1-self.smooth), tf.keras.utils.to_categorical(y_real_2, num_classes=5) * (1-self.smooth)]

class EncoderGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=8, dim=(128,128,128), n_channels=1, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.labels   = labels
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store sample
            image       = sitk.ReadImage(ID)
            image_array = np.asarray(sitk.GetArrayFromImage(image)).astype(float)
            X[i,] = (np.expand_dims(image_array, axis=-1) / np.percentile(image_array, 99.5)) - 0.5

            # Store class
            y[i] = self.labels[ID]

        return X, tf.keras.utils.to_categorical(y, num_classes=4)

class OCNNGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=8, dim=(128,128,128), n_channels=1, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store sample
            image       = sitk.ReadImage(ID)
            image_array = np.asarray(sitk.GetArrayFromImage(image)).astype(float)
            X[i,] = (np.expand_dims(image_array, axis=-1) / np.percentile(image_array, 99.5)) - 0.5

        return X

