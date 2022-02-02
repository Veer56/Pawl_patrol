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

    def train_combined(self, train_tabular, val_tabular):
        '''
        This method MOET NOG GETEST WORDEN
        '''

        # Use ImageDataGenerator to preprocess the samples or apply data augmentation
        datagen = ImageDataGenerator(**preprocess, **augment)

        # Use train_test_split to split the data into a training set and a validation set
        train_x, val_x, train_y, val_y = train_test_split(self.data_x, self.data_y, test_size=0.2, random_state=11)

        # Compute the mean of the training data
        datagen.fit(train_x)

        train_iterator = datagen.flow((train_x, train_tabular), train_y)
        test_iterator = datagen.flow((val_x, val_tabular) val_y)

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





test_model = models.Sequential()
test_model.add(layers.Dense(12, input_dim=1, activation='relu'))
test_model.add(layers.Dense(8, activation='relu'))
test_model.add(layers.Dense(1, activation='sigmoid'))

test_train_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
test_train_x = np.array(test_train_x)
test_train_y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
test_train_y = np.array(test_train_y)
test_epochs = 20
test_optimizer='adam'

test_preprocess = {'featurewise_center': True, 'featurewise_std_normalization' : True}

test_folds = 5

test = Train_and_evaluate(test_model, test_optimizer, test_train_x, test_train_y,  test_epochs)
test.train_kfold(test_folds)

#####
    # # Use compile to compile the model
    # model.compile(loss='mse', optimizer='adam', metrics=['RootMeanSquaredError'])
    #
    # # Use ImageDataGenerator to preprocess the samples or apply data augmentation
    # train_gen = preprocessing.image.ImageDataGenerator(**preprocess, **augment)
    # train_gen.fit(train_x)
    #
    # val_gen = preprocessing.image.ImageDataGenerator(**preprocess)
    # val_gen.fit(train_x)
    #
    # # Fit the model using the train and validation data
    # history = model.fit(train_gen.flow(train_x, train_y), epochs=epochs,
    #                     validation_data=val_gen.flow(val_x, val_y))
    #
    # # Make the figure for two plots
    # fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    #
    # # Plot the learning curves (loss and root mean squared error) of the train and validation
    # for i, metric in enumerate(['loss', 'root_mean_squared_error']):
    #     axs[i].plot(history.history[metric])
    #     axs[i].plot(history.history['val_' + metric])
    #     axs[i].legend(['training', 'validation'], loc='best')
    #
    #     axs[i].set_title('Model '+metric)
    #     axs[i].set_ylabel(metric)
    #     axs[i].set_xlabel('epoch')
    #
    # plt.show()
    #
    # # Display the validation root mean squared error after training
    # print(f"The validation root mean squared error: {model.evaluate(val_gen.flow(val_x, val_y))[1]}")
