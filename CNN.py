import random

import numpy as np
import pandas as pd
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D, Embedding, Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils

path_to_train_data = 'news_train.csv'
path_to_test_data = 'news_test.csv'

index2labels = {0: 'Reuters', 1: 'New York Times', 2: 'New York Post', 3: 'Breitbart'}
labels2index = {'Reuters': 0, 'New York Times': 1, 'New York Post': 2, 'Breitbart': 3}


def line_to_seq(line):
    '''
    Function to convert lines into character sequences
    :param line: input line
    :return: output character seq
    '''
    chars = list(line)
    line_chars_indices = list(map(lambda char: char_to_index[char], chars))
    return sequence.pad_sequences([line_chars_indices])[0]



def load_dataset(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    features_file = np.hstack((train.values[:, 1], test.values[:, 1]))
    char_list = list(set(''.join(features_file)))
    char_to_index = dict((c, i) for i, c in enumerate(char_list))
    index_to_char = dict((i, c) for i, c in enumerate(char_list))
    return char_to_index, index_to_char


char2index, index2char = load_dataset(path_to_train_data, path_to_test_data)


def load_dataset(input_path):
    df = pd.read_csv(input_path)
    labels = df.values[:, 0]
    features = df.values[:, 1]
    return features, labels

def process_train_data(input_tensor):
    characterized_list = []
    for index, value in np.ndenumerate(input_tensor):
        characterized_list.append([' '.join(value).split()])
    # characterized_list = [char2index[item] for item in characterized_list]
    indexes_list=[]
    for doc in characterized_list:
        for char in doc:
            indexes_list.append(char2index[char])
    return indexes_list

a = np.asarray(('adel','from iran','doesnt like'))
print(process_train_data(a))
def load_batches(input_path, batch_size):
    df = pd.read_csv(input_path)
    labels = df.values[:, 0]
    features = df.values[:, 1]
    i = 0
    while True:
        feature_batch = process_train_data([features[i:batch_size]])
        label_batch = [labels[i:batch_size]]
        yield feature_batch, label_batch

data_gen = load_batches(path_to_train_data, 2)
print(len(next(data_gen)[0]))
print(len(next(data_gen)[0]))
# loading the files
path_to_train_data = 'news_train.csv'
read_train_file, labels_in_train = load_dataset(path_to_train_data)
read_train_file = np.ndarray.tolist(read_train_file)
labels_in_train = np.ndarray.tolist(labels_in_train)
# read_train_file = [line.rstrip('\n') for line in open('x.txt')]
# read_test_file = [line.rstrip('\n') for line in open('sample.txt')]
path_to_test_data = 'news_test.csv'
read_test_file, labels_in_test = load_dataset(path_to_test_data)
read_test_file = np.ndarray.tolist(read_test_file)

# Test
# labels_in_train = [line.rstrip('\n') for line in open('y.txt')]
index_to_label = {i: k for i, k in enumerate(labels_in_train)}
label_to_index = {k: i for i, k in enumerate(labels_in_train)}
del read_train_file, labels_in_train, read_test_file, labels_in_test
# compile a list of characters
char_list = list(set(''.join(read_train_file + read_test_file)))
labels_list = list(index_to_label.keys())

# convert characters to indices
char_to_index = dict((c, i) for i, c in enumerate(char_list))
index_to_char = dict((i, c) for i, c in enumerate(char_list))

# Get longest string
max_len = 0
for l in (read_train_file + read_test_file):
    if len(l) > max_len:
        max_len = len(l)

# Initialize input and label variables
x = []
y = []
X_predict = []

# Convert line by line
for line_, label_ in zip(read_train_file, index_to_label):
    x.append(line_to_seq(line_))
    y.append(label_)

for line_ in read_test_file:
    X_predict.append(line_to_seq(line_))

x = np.array(x).astype(np.uint8)
y = np_utils.to_categorical(np.array(y)).astype(np.bool)
X_predict = np.array(X_predict).astype(np.uint8)

print('shape_x: ', x.shape, 'shape_y: ', y.shape)
print("number of labels: ", len(labels_list))

X_train = x
y_train = y

# defining hyperparameters
batch_size = 100
epochs = 100

# defining model
model = Sequential()
# input layer
model.add(Embedding(batch_size, None, mask_zero=False))

# Convolution layer
model.add(Conv1D(1024, 7, activation='relu', name='activation_1_conv1d'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))

# FC
# Generally We try to avoid/lower FC connections because they only add
# tremendous amount parameter and hence consume
#  memory but do not add much to the network learning
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# softmax layer to predict each label
model.add(Dense(len(labels_list)))
model.add(Activation('softmax'))

# RMSprop optimizer
optimizer = optimizers.RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
early_stopping = EarlyStopping(patience=10, verbose=1)

# checkpoint saver
checkpointer = ModelCheckpoint(filepath='code_table_model.hdf5',
                               verbose=1,
                               save_best_only=True)

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          validation_split=0.2,
          callbacks=[early_stopping, checkpointer])

y_predict = model.predict_classes(X_predict)

for test, pred in zip(read_test_file, y_predict):
    print("label:", test, 'prediction: ', index_to_label[pred])
