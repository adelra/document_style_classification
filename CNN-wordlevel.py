import numpy as np
import pandas as pd
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D, Embedding, Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing import sequence

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
    words = list(line)
    line_words_indices = list(map(lambda word: word_to_index[word], words))
    return sequence.pad_sequences([line_words_indices])[0]


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


def tensor_to_index(tensor):
    processed_tensor = list(map(word2index.get, tensor))
    return processed_tensor


def load_batches(input_path, batch_size):
    df = pd.read_csv(input_path)
    labels = df.values[:, 0]
    features = df.values[:, 1]
    i = 0
    while True:
        feature_batch = features[i:batch_size]
        label_batch = labels[i:batch_size]
        label_batch = list(map(labels2index.get, label_batch))
        feature_batch = np.ndarray.tolist(feature_batch)
        for index, line in enumerate(feature_batch):
            feature_batch[index] = tensor_to_index(line.split())
        return feature_batch


# defining hyperparameters
batch_size = 100
epochs = 100

# defining model
model = Sequential()
# input layer
max_len = len(max(open(path_to_train_data, 'r'), key=len))
model.add(Embedding(batch_size, 32, mask_zero=False))

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
model.add(Dense(4))
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

model.fit_generator(load_batches(path_to_train_data, batch_size=batch_size), epochs=epochs,
                    callbacks=[early_stopping, checkpointer])

# y_predict = model.predict_classes(X_predict)

# for test, pred in zip(read_test_file, y_predict):
#     print("label:", test, 'prediction: ', index_to_label[pred])
