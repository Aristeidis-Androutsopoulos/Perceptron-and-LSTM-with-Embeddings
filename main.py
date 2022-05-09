# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
import re
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import keras_tuner as kt
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.ticker as mticker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@dataclass
class Model_Config:
    # SECTION: Model Configuration

    # SECTION: A.2
    # Model configuration
    # NOTE: variables for plot control and naming
    # TODO: Change values accordingly in each run

    section: str = 'A4'
    distinction: str = '3'

    folder: str = 'plots\\'

    # Variable for Hyperparameter Autotune
    tuning: bool = False
    # TODO: Change batch size to 128 for the 20 model

    batch_size: int = 64
    verbose: int = 2
    no_epochs: int = 100

    num_folds: int = 5

    # SECTION: A.2.1 - Loss Function and Metrics
    # Binary cross entropy is better suited for multilabel classification
    loss_function: str = 'binary_crossentropy'
    # loss_function: str = 'mean_squared_error'
    metrics: list = field(default_factory=lambda: ['accuracy', 'mean_squared_error'])
    # metrics: list = field(default_factory=lambda: ['categorical_accuracy', 'binary_crossentropy'])

    # SECTION: A.2.2 - Number of output neurons
    no_classes: int = 20

    # SECTION: A.2.3 - Activation Function of Inner Layers
    # Default Choice, Seek report for more
    activation_function_inner: str = 'relu'

    # SECTION: A.2.4 - Activation Function of Output Layer
    # Sigmoid allows multilabel classification
    activation_function_output: str = 'sigmoid'

    # SECTION: A.2.5 - # of Neurons in 1st Hidden Layer
    # TODO - 20, 4270, 8540, change in each run
    hidden1_num_of_neurons: int = 4270

    # SECTION: A.2.6 - # of Neurons in 2nd Hidden Layer
    # TODO - 2135, 4270, 8540, change in each run
    hidden2_num_of_neurons: int = 2135

    # SECTION: A.2.7 - Early Stopping
    # TODO - Find good min_delta and good patience for this specific task
    #   problem with hours of run. Use larger min delta
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.003, patience=10,
                                                      restore_best_weights=True)

    # SECTION: A.3 - Hyperparameter Tuning on Learning Rate and Momentum
    # TODO: Learning Rate, Initially 0.001. - 0.001, 0.05, 0.1
    learning_rate: float = 0.001

    # TODO: Momentum - 0.2, 0.6
    momentum: float = 0.6

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=False)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # SECTION: A.4 - Regularisation with L2
    # TODO: Weight Decay - 0.1, 0.5, 0.9
    weight_decay: float = 0.9

    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    bias_regularizer = tf.keras.regularizers.l2(weight_decay)

    # NOTE: EXTRA - Kernel and Bias Initializer
    # TODO: Initialize to smart distributions
    # NOTE: He Normalization is a great choice for RELU activation function
    # Carefully to use He Uniform and not He Normal. He uniform gives all the kernel weights the same value initially
    kernel_initializer = tf.keras.initializers.HeUniform()
    bias_initializer = tf.keras.initializers.Zeros()


# opening the file in read mode

def preprocess():
    # SECTION: Preprocess of data
    # open files and save data into lists
    open_file = open("test-data.dat", "r")
    test_data = open_file.read()
    test_data_into_list = test_data.split("\n")
    open_file.close()

    open_file = open("train-data.dat", "r")
    train_data = open_file.read()
    train_data_into_list = train_data.split("\n")
    open_file.close()

    # Throw last value because its empty
    train_data_into_list.pop()
    test_data_into_list.pop()

    # Combine all data into one set for the 5-fold validation
    data_into_list = train_data_into_list + test_data_into_list

    # Remove all values of type <int>
    lst_of_data = [re.sub('<[^>]+>', '', x) for x in data_into_list]

    # Set our vocabulary to the right length
    vocab_range = np.arange(0, 8520, 1)
    vocab_range = vocab_range.tolist()
    vocab_output = [str(x) for x in vocab_range]

    # Vectorize data using Scikit Learn for each number in our vocabulary
    vectorizer = CountVectorizer(tokenizer=lambda txt: txt.split())
    vectorizer.fit_transform(vocab_output)
    k_hot_end = vectorizer.transform(lst_of_data)
    df_bow_sklearn = pd.DataFrame(k_hot_end.toarray(), columns=vectorizer.get_feature_names_out())
    end_form = df_bow_sklearn.to_numpy()

    # Load all labels and carefully combine them into one in the order of the data combination above
    test_labels = np.loadtxt("test-label.dat", dtype=int)
    train_labels = np.loadtxt("train-label.dat", dtype=int)

    labels = np.vstack((train_labels, test_labels))

    # SECTION: Normalise or Standardise
    # NOTE: Preprocess data with some Variability or by Centering them and projecting them to a distribution

    # fit scaler on training data
    # norm = MinMaxScaler()
    #
    # # transform training data
    # x_train_norm = norm.fit_transform(my_new)
    #
    # # transform testing data
    # label_data = norm.fit_transform(con)

    # NOTE: Standardisation - In use. In practice better performance was seen while standardize than normalised.
    #   Also, no standardization for the output data. Will set a distribution to non distributed data. AVOID!
    # TODO: Standardize in each column so that we don't assign a distribution to categorical features

    stand = StandardScaler()
    # stand = MaxAbsScaler()
    stand_data = stand.fit_transform(end_form.T).T

    # stand_label = stand.fit_transform(con.T).T

    # X_minmax = minmax_scale.transform(x.T).T

    return stand_data, labels
    # return x_train_norm, label_data
    # return stand_data, stand_label


def model_builder(hp):
    # SECTION: Tuning model build
    # NOTE: Special config for this model builder, because of keras initialized function.
    config = Model_Config()

    # TODO: Set the same model through advanced usage of model_builder extention with keras
    # Make the model the same for the tuning purposes. Further Explanation in actual model bellow
    model = tf.keras.models.Sequential([

        tf.keras.layers.Input(shape=(8520,)),

        tf.keras.layers.Dense(config.hidden1_num_of_neurons, activation=config.activation_function_inner),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(config.no_classes, activation=config.activation_function_output)

    ])

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-2, 1e-1])

    hp_momentum = hp.Choice('momentum', values=[0.2, 0.6])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=hp_learning_rate, momentum=hp_momentum, nesterov=True,
                                                    name="SGD"),
                  loss=config.loss_function,
                  metrics=config.metrics)

    return model


# TODO: Set the function with no repeatable code. See example in https://keras.io/guides/keras_tuner/getting_started/
def Tune_Model(data, labels):
    # SECTION: Tune Model

    # Tune model with exhaustive random search to check all possible combinations of both learning rate and momentum
    tuner = kt.RandomSearch(model_builder,
                            objective='loss',
                            directory='pythonProject3',
                            project_name='Tuning')

    # Early stopping with relatively small patience but 0 value difference
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Set tuner to see all search space
    tuner.search_space_summary()

    # Initiate the Search
    tuner.search(data, labels, epochs=10, validation_split=0.2, callbacks=[stop_early], verbose=2)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Print them in orderly fashion

    print(f"""
    The hyperparameter search is complete. The optimal momentum for the optimizer is {best_hps.get('momentum')} and 
    the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}. 
    """)

    return best_hps.get('learning_rate'), best_hps.get('momentum')


def neural_model(config, data, labels):
    # SECTION: Keras Model

    # NOTE: Define the K-fold Cross Validator
    kfold = KFold(n_splits=config.num_folds, shuffle=True)
    # Lists for saving metrics for every fold
    all_fold_mse = list()
    all_fold_ce = list()
    all_fold_acc = list()

    # For loop for Cross Validation
    # NOTE: Checking the fold for better results
    for train, test in kfold.split(data, labels):
        # NOTE: MODEL with type sequential using the sequential function initializer

        model = tf.keras.models.Sequential([

            # NOTE: Input Layer
            tf.keras.layers.Input(shape=(8520,)),

            # NOTE: First Hidden Layer with bias and kernel initializers
            tf.keras.layers.Dense(config.hidden1_num_of_neurons, activation=config.activation_function_inner,
                                  kernel_initializer=config.kernel_initializer,
                                  bias_initializer=config.bias_initializer,
                                  kernel_regularizer=config.kernel_regularizer,
                                  bias_regularizer=config.bias_regularizer),

            # Drop out Layer as an overfitting solution
            tf.keras.layers.Dropout(0.2),

            # NOTE: Second Hidden Layer with bias and kernel initializers
            tf.keras.layers.Dense(config.hidden2_num_of_neurons, activation=config.activation_function_inner,
                                  kernel_initializer=config.kernel_initializer,
                                  bias_initializer=config.bias_initializer,
                                  kernel_regularizer=config.kernel_regularizer,
                                  bias_regularizer=config.bias_regularizer),

            # TODO: Set Dropout layers if model converges - DONE
            tf.keras.layers.Dropout(0.2),

            # NOTE: Output Layer with bias and kernel initializers
            tf.keras.layers.Dense(config.no_classes, activation=config.activation_function_output,
                                  kernel_initializer=config.kernel_initializer,
                                  bias_initializer=config.bias_initializer,
                                  kernel_regularizer=config.kernel_regularizer,
                                  bias_regularizer=config.bias_regularizer)

        ])

        # NOTE: Model Compile
        model.compile(optimizer=config.optimizer,
                      loss=config.loss_function,
                      metrics=config.metrics)

        # NOTE: Model Fit
        # TODO: Find possible alterations with better results for the occurring over-fit problem

        history = model.fit(data[train], labels[train], epochs=config.no_epochs, batch_size=config.batch_size,
                            callbacks=[config.early_stopping], validation_split=0.2, verbose=config.verbose)

        # print(model.metrics_names)

        # NOTE: Model Evaluate
        ce_loss, accuracy, mse = model.evaluate(data[test], labels[test])

        # Keep the greatest model's history
        if len(all_fold_ce) == 0:
            best_fold_history = history
        elif ce_loss < all_fold_ce[-1]:
            best_fold_history = history

        # Save each fold's loss and metrics
        all_fold_ce.append(ce_loss)
        all_fold_mse.append(mse)
        all_fold_acc.append(accuracy)

        # Print message in each fold with mean values
        print("Mean of Cross-Entropy loss is: ", np.mean(all_fold_ce))
        print("Mean of Mean Squared Error is: ", np.mean(all_fold_mse))
        print("Mean of Accuracy is: ", np.mean(all_fold_acc))

    # NOTE: Used to save mean of all folds for report
    #   empty when uploaded for potential run by other person
    f = open("numbers.txt", "a")
    f.write("Start \n")
    f.write("Run " + config.section + ', ' + config.distinction + '\n')
    f.write(str(np.mean(all_fold_ce)) + " , ")
    f.write(str(np.mean(all_fold_mse)) + " , ")
    f.write(str(np.mean(all_fold_acc)) + "\n")
    f.close()

    # return the best history
    # NOTE: Ignore warning, made sure it always has a history to return
    return best_fold_history


def plots_run_time(history, config):
    # SECTION: Plotting

    # Plot Cross Entropy by itself
    # plt.style.use('_mpl-gallery')
    sns.set()
    sns.set_theme()
    sns.color_palette("Paired")
    sns.set_style("whitegrid")
    fig1, ax0 = plt.subplots(constrained_layout=True, figsize=(10, 5))

    # ax0.set_facecolor('0.8')
    ax0.plot(history.history['loss'], label='Cross Entropy (Train)', color="#003f5c")
    ax0.plot(history.history['val_loss'], label='Validation Cross Entropy (Test)', color="#ffa600")
    ax0.set_xlabel('Epochs')
    ax0.set_ylabel('Error')
    ax0.set_title('Cross Entropy Evaluation, ' + config.section + ', ' + config.distinction, loc='center')
    ax0.legend()

    # NOTE: for small # epochs uncomment bellow
    # ax0.xaxis.set_major_locator(mticker.MultipleLocator(1))

    # save the figure to file
    plt.savefig(config.folder + config.section + '_ce_' + config.distinction + '.png', format='png', pad_inches=0.2,
                bbox_inches="tight")

    plt.show()

    # Plot Accuracy and MSE together
    fig2, [ax1, ax2] = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))

    # ax1.set_facecolor('0.8')
    ax1.plot(history.history['accuracy'], label='Accuracy (Train)', color="#003f5c")
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy (Test)', color="#ffa600")
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Error')
    ax1.set_title('Categorical Accuracy Metric, ' + config.section + ', ' + config.distinction, loc='center')
    ax1.legend()

    # NOTE: for small # epochs uncomment bellow
    # ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))

    # ax2.set_facecolor('0.8')
    ax2.plot(history.history['mean_squared_error'], label='MSE (Train)', color="#003f5c")
    ax2.plot(history.history['val_mean_squared_error'], label='Validation MSE (Test)', color="#ffa600")
    ax2.set_xlabel('Epochs')
    ax2.set_title('MSE Metric, ' + config.section + ', ' + config.distinction, loc='center')
    ax2.legend()

    # NOTE: for small # epochs uncomment bellow
    # ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))

    # save the figure to file
    plt.savefig(config.folder + config.section + '_metrics_' + config.distinction + '.png', format='png',
                pad_inches=0.2,
                bbox_inches="tight")

    plt.show()

    print('plot done')


if __name__ == '__main__':
    # Preprocess data by calling Preprocess()
    my_data, my_labels = preprocess()

    # Initialise configuration of the model with last changes
    model_config = Model_Config()

    # NOTE: If tuning automatically is needed uncomment next line (should also comment previous line)
    # model_config = Model_Config(tuning=True)

    if model_config.tuning:
        my_model = Tune_Model(my_data, my_labels)

    # Run the Model and return the best history out of all folds
    best_fold_history = neural_model(model_config, my_data, my_labels)

    # Plot everything neede according to history returned
    plots_run_time(best_fold_history, model_config)

    # Print DONE for ending
    print("done")
