from tensorflow.keras import layers, models, preprocessing, losses, metrics
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class Train_and_evaluate():
    '''
    This class compiles a model on samples (train_x) and their
    labels (train_y), using the Mean Squared Error loss function and a chosen optimizer.
    Additionally, it evaluates the performance of the model on validation data
    (val_x and val_y) using the Root Mean Squared Error. It is possible to pass arguments
    for preprocessing of samples and data augmentation.
    '''

    def __init__(self, model, optimizer, data_x, data_y, epochs):
        '''
        This method initializes the model, the chosen optimizer, the features (data_x)
        and the target values (data_y).
        '''

        self.model = model
        self.optimizer = optimizer
        self.data_x = data_x
        self.data_y = data_y
        self.epochs = epochs


    def train_regular(self, preprocess={}, augment={}):
        '''
        This method trains a model and evaluates its performance. It plots the
        learning curves. Additionally, it is possible to pass arguments
        for preprocessing of samples and data augmentation.
        '''

        # Use ImageDataGenerator to preprocess the samples or apply data augmentation
        datagen = ImageDataGenerator(**preprocess, **augment)

        # Use train_test_split to split the data into a training set and a validation set
        train_x, val_x, train_y, val_y = train_test_split(self.data_x, self.data_y, test_size=0.2, random_state=11)

        # Compute the mean of the training data
        datagen.fit(train_x)

        # Wat?
        train_iterator = datagen.flow(train_x, train_y)
        test_iterator = datagen.flow(val_x, val_y)

        # Use compile to compile the model
        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=['RootMeanSquaredError'])

        # Fit the model using the train and validation data
        history = self.model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator),
                    epochs=self.epochs)

        # Plot the learning curves
        plt.plot(history.history['RootMeanSquaredError'])
        plt.plot(history.history['val_RootMeanSquaredError'])
        plt.title('model RootMeanSquaredError')
        plt.ylabel('RootMeanSquaredError')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.show()


    def train_kfold(self, folds, preprocess={}, augment={}):
        '''
        This method trains a model and evaluates its performance of the model on
        validation data (val_x and val_y) using the Root Mean Squared Error.
        Furthermore, K_fold cross_validation is implemented for a given number
        of folds. It is possible to pass arguments for preprocessing of samples
        and data augmentation.
        '''

        kfold = KFold(n_splits=folds)

        # Define an empty list to store the cv scores of every fold
        cv_scores= []

        # Create a for_loop to fit the model for every fold
        for train, test in kfold.split(self.data_x, self.data_y):

            # Use ImageDataGenerator to preprocess the samples or apply data augmentation
            datagen = ImageDataGenerator(**preprocess, **augment)

            # Use train_test_split to split the data into a training set and a validation set
            train_x, val_x, train_y, val_y = train_test_split(self.data_x, self.data_y, test_size=0.2, random_state=11)

            # Compute the mean of the training data
            datagen.fit(train_x)


            train_iterator = datagen.flow(train_x, train_y)
            test_iterator = datagen.flow(val_x, val_y)

            # Fit the model using compile
            self.model.compile(loss='mse', optimizer=self.optimizer, metrics=['RootMeanSquaredError'])

            # Fit the model using the train and validation data
            history = self.model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator),
                        epochs=self.epochs)

        try:
            for i, metric in enumerate(['loss', 'root_mean_squared_error']):
                axs[i].plot(history.history[metric])
                axs[i].plot(history.history['val'+metric])
                axs[i].legend(['training', 'validation'], loc='best')

                axs[i].set_title('Model '+metric)
                axs[i].set_ylabel(metric)
                axs[i].setxlabel('epoch')
                
        except:
            for i, metric in enumerate(['loss', 'RootMeanSquaredError']):
                axs[i].plot(history.history[metric])
                axs[i].plot(history.history['val'+metric])
                axs[i].legend(['training', 'validation'], loc='best')

                axs[i].set_title('Model '+metric)
                axs[i].set_ylabel(metric)
                axs[i].set_xlabel('epoch')

    def train_combined(self, tabular_data, preprocess={}, augment={}):
        '''
        This method takes data regarding training samples and validation
        samples for both image data as well as tabular data, and compiles the model
        using the Mean Squared Error loss function and the Adam optimizer. It evaluates the performance of
        the model using the Root Mean Squared Error. It is possible to pass arguments
        for preprocessing of samples and data augmentation.
        '''

        # Use ImageDataGenerator to preprocess the samples or apply data augmentation
        datagen = ImageDataGenerator(**preprocess, **augment)

        # Use train_test_split to split the data into a training set and a validation set
        train_x, val_x, train_y, val_y, train_tabular, test_tabular = train_test_split(self.data_x, self.data_y, tabular_data, test_size=0.2, random_state=11)

        train_gen = ImageDataGenerator(**preprocess, **augment)
        train_gen.fit(train_x)

        val_gen = ImageDataGenerator(**preprocess, **augment)
        val_gen.fit(val_x)

        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=['RootMeanSquaredError'])

        history = model.fit(train_gen.flow((train_x, train_tabular), train_y), epochs=self.epochs,
                            validation_data=val_gen.flow((val_x, test_tabular), val_y))

        # Plot the learning curves
        plt.plot(history.history['RootMeanSquaredError'])
        plt.plot(history.history['val_RootMeanSquaredError'])
        plt.title('model RootMeanSquaredError')
        plt.ylabel('RootMeanSquaredError')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.show()
