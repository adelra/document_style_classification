import numpy as np
import pandas as pd
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.utils import np_utils

path_to_train_data = 'news_train.csv'
path_to_test_data = 'news_test.csv'

index2labels = {0: 'Reuters', 1: 'New York Times', 2: 'New York Post', 3: 'Breitbart'}
labels2index = {'Reuters': 0, 'New York Times': 1, 'New York Post': 2, 'Breitbart': 3}

max_len = len(max(open(path_to_train_data, 'r'), key=len))
vocab_size = len(set(open(path_to_train_data, 'r').read().split()))


def load_dataset(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    features_file = np.hstack((train.values[:, 1], test.values[:, 1]))
    features_file = np.ndarray.tolist(features_file)
    all_words = []
    for item in features_file:
        words = item.split()
        words = set(words)
        all_words.append(words)
    all_words = set([item for sublist in all_words for item in sublist])
    # all_words = [set(i) for i in all_words]
    words_to_index = dict((c, i) for i, c in enumerate(all_words))
    index_to_words = dict((i, c) for i, c in enumerate(all_words))
    return words_to_index, index_to_words


word2index, index2word = load_dataset(path_to_train_data, path_to_test_data)


def load_batches(input_path, batch_size):
    '''
    The main generator function to load and batchify the data
    :param input_path: input path for the training/testing data
    :param batch_size: number of batches
    :return: tuple of the features and labels
    '''
    df = pd.read_csv(input_path)
    labels = df.values[:, 0]
    features = df.values[:, 1]
    global sequence_length
    sequence_length = len(features)
    start = 0
    end = batch_size
    while True:
        while end <= len(features):
            x_batch = np.zeros([batch_size, max_len])
            # y_batch = np.zeros()
            for i in range(batch_size):
                doc_to_index = list(map(word2index.get, features[i + start].split()))
                for index, value in enumerate(doc_to_index):
                    x_batch[i, index] = value
            y_batch = labels[start:end]
            y_batch = list(map(labels2index.get, y_batch))
            y_batch = np_utils.to_categorical(y_batch, num_classes=4)
            x_batch = x_batch.reshape([batch_size, max_len, 1])
            y_batch = y_batch.reshape([batch_size, 4])
            start += batch_size
            end += batch_size
            yield (x_batch, y_batch)


# defining hyperparameters
batch_size = 300
epochs = 100
print('input shape: ', next(load_batches(path_to_train_data, batch_size))[0].shape)

# Convolution layer
model = Sequential()
model.add(Conv1D(128, kernel_size=3, strides=1,
                 activation='relu',
                 input_shape=(max_len, 1), padding='SAME'))
model.add(MaxPooling1D(pool_size=3, strides=2))
model.add(Conv1D(256, 5, activation='relu', padding='SAME'))
model.add(MaxPooling1D(pool_size=2))

# FC layer
# before the FC the we flatten our tensors
# Generally We try to avoid/lower FC connections because they only add
# tremendous amount parameter and hence consume
#  memory but do not add much to the network learning

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(4, activation='softmax', name='softmax'))
# RMSprop optimizer
optimizer = optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
early_stopping = EarlyStopping(patience=10, verbose=1)
steps_per_epoch = (sequence_length / batch_size)
# checkpoint saver
checkpointer = ModelCheckpoint(filepath='code_table_model.hdf5',
                               verbose=1,
                               save_best_only=True)
batch_iterator = load_batches(path_to_train_data, batch_size=batch_size)
model.fit_generator(batch_iterator, epochs=epochs,
                    callbacks=[early_stopping, checkpointer], steps_per_epoch=steps_per_epoch)

# y_predict = model.predict_classes(X_predict)

# for test, pred in zip(read_test_file, y_predict):
#     print("label:", test, 'prediction: ', index_to_label[pred])
