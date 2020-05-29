#!/usr/bin/env python3
# coding: utf8

import pickle
import numpy as np

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def load_cifar_10_data(negatives=False):
    """
    Return train_data, train_filenames, train_labels, 
    test_data, test_filenames, test_labels
    """

    # Get Training Data
    cifar_train_data = None
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle("cifar-10-batches-py/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_labels += cifar_train_data_dict[b'labels']

    # Reshape data based on 3 color channels
    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)

    # Save labels
    cifar_train_labels = np.array(cifar_train_labels)
    cifar_train_labels = cifar_train_labels.reshape(len(cifar_train_data),1)

    # Get test data
    cifar_test_data_dict = unpickle("cifar-10-batches-py/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)

    # Save labels
    cifar_test_labels = np.array(cifar_test_labels)
    cifar_test_labels = cifar_test_labels.reshape(len(cifar_test_data),1)

    return cifar_train_data, cifar_train_labels, cifar_test_data, cifar_test_labels


if __name__ == "__main__":

    # Load train and test data
    train_data, train_labels, test_data, test_labels = load_cifar_10_data()

    print("Train data: ", train_data.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test labels: ", test_labels.shape)
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    history = model.fit(train_data, train_labels, epochs=10, 
                    validation_data=(test_data, test_labels))

    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
    print(test_acc)