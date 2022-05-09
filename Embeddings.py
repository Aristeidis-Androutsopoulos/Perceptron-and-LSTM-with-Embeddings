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
from keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@dataclass
class Model_Config:
    # SECTION: Model Configuration
    max_length: int = 256
    # SECTION: A.2
    # Model configuration
    # NOTE: variables for plot control and naming
    # TODO: Change values accordingly

    section: str = 'A5.1'
    distinction: str = 'embeddings_1'

    folder: str = 'plots\\'

    tuning: bool = False
    # TODO: Change batch size to 128 for the 20 model

    batch_size: int = 64
    verbose: int = 2
    no_epochs: int = 50

    num_folds: int = 5

    # SECTION: A.2.1 - Loss Function and Metrics
    loss_function: str = 'binary_crossentropy'
    # loss_function: str = 'mean_squared_error'
    metrics: list = field(default_factory=lambda: ['accuracy', 'mean_squared_error'])
    # metrics: list = field(default_factory=lambda: ['categorical_accuracy', 'binary_crossentropy'])

    # SECTION: A.2.2 - Number of output neurons
    no_classes: int = 20

    # SECTION: A.2.3 - Activation Function of Inner Layers
    activation_function_inner: str = 'relu'

    # SECTION: A.2.4 - Activation Function of Output Layer
    activation_function_output: str = 'sigmoid'

    # SECTION: A.2.5 - # of Neurons in 1st Hidden Layer
    # TODO - 20, 4270, 8540
    hidden1_num_of_neurons: int = 4270

    # SECTION: A.2.6 - # of Neurons in 2nd Hidden Layer
    # TODO - 2135, 4270, 8540
    hidden2_num_of_neurons: int = 2135

    # SECTION: A.2.7 - Early Stopping
    # TODO - Find good min_delta and good patience
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=10,
                                                      restore_best_weights=True)

    # SECTION: A.3 - Hyperparameter Tuning on Learning Rate and Momentum
    # TODO: Learning Rate, Initially 0.001. - 0.001, 0.05, 0.1
    learning_rate: float = 0.001

    # TODO: Momentum - 0.2, 0.6
    momentum: float = 0

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=False)

    # SECTION: A.4 - Regularisation with L2
    # TODO: Weight Decay - 0.1, 0.5, 0.9
    weight_decay: float = 0.1

    # NOTE: EXTRA - Kernel and Bias Initializer
    # TODO: Initialize to smart distributions
    # NOTE: He Normalization is a great choice for RELU activation function
    kernel_initializer = tf.keras.initializers.HeUniform()
    bias_initializer = tf.keras.initializers.Zeros()


# opening the file in read mode

def preprocess(config):
    # SECTION: Preprocess of data
    open_file = open("test-data.dat", "r")
    test_data = open_file.read()
    test_data_into_list = test_data.split("\n")
    open_file.close()

    open_file = open("train-data.dat", "r")
    train_data = open_file.read()
    train_data_into_list = train_data.split("\n")
    open_file.close()

    train_data_into_list.pop()
    test_data_into_list.pop()

    data_into_list = train_data_into_list + test_data_into_list

    lst = [re.sub('<[^>]+>', '', x) for x in data_into_list]

    vocab_range = np.arange(0, 8520, 1)
    vocab_range = vocab_range.tolist()
    output = [str(x) for x in vocab_range]

    # vectorizer = CountVectorizer(tokenizer=lambda txt: txt.split())
    # vectorizer.fit_transform(output)
    # k_hot_end = vectorizer.transform(lst)
    # df_bow_sklearn = pd.DataFrame(k_hot_end.toarray(), columns=vectorizer.get_feature_names_out())
    # my_new = df_bow_sklearn.to_numpy()

    test_labels = np.loadtxt("test-label.dat", dtype=int)
    train_labels = np.loadtxt("train-label.dat", dtype=int)

    con = np.vstack((train_labels, test_labels))

    last_list = list()

    for i in lst:
        a_list = i.split()
        map_object = map(int, a_list)
        list_of_integers = list(map_object)
        love = map(lambda x: x + 1, list_of_integers)
        last_list.append(list(love))

    padded_docs = pad_sequences(last_list, maxlen=config.max_length, padding='post')
    print(padded_docs)

    # NOTE: No Standardization or Normalization before the embedding layer

    return padded_docs, con
    # return stand_data, stand_label


def model_builder(hp):
    config = Model_Config()

    model = tf.keras.models.Sequential([

        tf.keras.layers.Input(shape=(8521,)),

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
    tuner = kt.RandomSearch(model_builder,
                            objective='loss',
                            directory='pythonProject3',
                            project_name='Tuning')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search_space_summary()

    tuner.search(data, labels, epochs=10, validation_split=0.2, callbacks=[stop_early], verbose=2)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

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
    # TODO: !!!!! !!!!! !!!!! Change it
    # for train, test in kfold.split(data, labels):
    # NOTE: MODEL
    model = tf.keras.models.Sequential([

        # NOTE: Embedding Layer

        tf.keras.layers.Embedding(8521, 64, input_length=config.max_length, trainable=True, mask_zero=True),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dropout(0.1),

        # NOTE: First Hidden Layer
        tf.keras.layers.Dense(config.hidden1_num_of_neurons, activation=config.activation_function_inner,
                              kernel_initializer=config.kernel_initializer,
                              bias_initializer=config.bias_initializer),

        # NOTE: Second Hidden Layer
        # tf.keras.layers.Dense(config.hidden2_num_of_neurons, activation=config.activation_function_inner,
        #                       kernel_initializer=config.kernel_initializer,
        #                       bias_initializer=config.bias_initializer),

        # TODO: Set Dropout layers if model converges
        #  Only if Over-fitting happens
        tf.keras.layers.Dropout(0.1),

        # NOTE: Output Layer
        tf.keras.layers.Dense(config.no_classes, activation=config.activation_function_output,
                              kernel_initializer=config.kernel_initializer,
                              bias_initializer=config.bias_initializer)

    ])

    # NOTE: Model Compile
    model.compile(optimizer=config.optimizer,
                  loss=config.loss_function,
                  metrics=config.metrics)

    # NOTE: Model Fit
    # TODO: !!!!! !!!!! !!!!! Change it
    history = model.fit(data, labels, epochs=config.no_epochs, batch_size=config.batch_size,
                        callbacks=[config.early_stopping], validation_split=0.3, verbose=config.verbose)
    # history = model.fit(data[train], labels[train], epochs=config.no_epochs, batch_size=config.batch_size,
    #                     callbacks=[config.early_stopping], validation_split=0.2, verbose=config.verbose)
    print(model.metrics_names)

    # NOTE: Model Evaluate
    ce_loss, accuracy, mse = model.evaluate(data, labels)
    # ce_loss, accuracy, mse = model.evaluate(data[test], labels[test])

    # Keep the greatest model's history
    if len(all_fold_ce) == 0:
        best_fold_history = history
    elif ce_loss < all_fold_ce[-1]:
        best_fold_history = history

    # Save each fold's loss and metrics
    all_fold_ce.append(ce_loss)
    all_fold_mse.append(mse)
    all_fold_acc.append(accuracy)

    print("Mean of Cross-Entropy loss is: ", np.mean(all_fold_ce))
    print("Mean of Mean Squared Error is: ", np.mean(all_fold_mse))
    print("Mean of Accuracy is: ", np.mean(all_fold_acc))

    # NOTE: Used to save mean of all folds for report
    f = open("numbers.txt", "a")
    f.write("Start \n")
    f.write("Run " + config.section + ', ' + config.distinction + '\n')
    f.write(str(np.mean(all_fold_ce)) + " , ")
    f.write(str(np.mean(all_fold_mse)) + " , ")
    f.write(str(np.mean(all_fold_acc)) + "\n")
    f.close()

    return best_fold_history


def plots_run_time(history, config):
    # SECTION: Plotting

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
    plt.savefig(config.folder + config.section + '_ce_' + config.distinction + '.png', format='png', pad_inches=0.2,
                bbox_inches="tight")

    plt.show()

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
    plt.savefig(config.folder + config.section + '_metrics_' + config.distinction + '.png', format='png',
                pad_inches=0.2,
                bbox_inches="tight")

    plt.show()

    print('plot done')


if __name__ == '__main__':
    model_config = Model_Config()

    my_data, my_labels = preprocess(model_config)
    # NOTE: If tuning is needed uncomment next line (should also comment previous line)
    # model_config = Model_Config(tuning=True)

    if model_config.tuning == True:
        my_model = Tune_Model(my_data, my_labels)

    best_fold_history = neural_model(model_config, my_data, my_labels)

    plots_run_time(best_fold_history, model_config)

    print("done")
