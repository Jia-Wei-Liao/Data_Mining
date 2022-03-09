import os
import re
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
import gensim.downloader
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Dense, LSTM, Dropout,
                                     Conv1D, MaxPooling1D, Flatten)


def transforms(sentence):
    nltk.download('stopwords')
    words = re.sub('[^a-zA-Z]',' ', sentence.lower()).split()
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if not w in stop_words]

    return words


def lr_schedule(epoch, lr):
    if epoch % 5 == 4:
        lr *= 0.8

    return lr


def plot_acc(record):
    epochs = range(len(record.history['accuracy']))
    train_acc = record.history['accuracy']
    valid_acc = record.history['val_accuracy']

    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_acc, 'o-', label='training accuracy')
    plt.plot(epochs, valid_acc, 'o-', label='validation accuracy')
    plt.ylim([0.6, 0.9])
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    return None


def plot_loss(record):
    epochs = range(len(record.history['loss']))
    train_loss = record.history['loss']
    valid_loss = record.history['val_loss']

    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_loss, 'o-', label='training loss')
    plt.plot(epochs, valid_loss, 'o-', label='validation loss')
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend()
    plt.grid()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    return None


if __name__ == '__main__':
    df = pd.read_csv('yelp.csv')
    
    # Data Preprocessing
    data = pd.DataFrame([])
    data.loc[:, 'sentence'] = df['text'].map(transforms)
    data.loc[:, 'label'] = df['stars'].map(lambda x: int(x>=4))


    # Split training and test data
    X_train_word, X_test_word, y_train, y_test = train_test_split(data.sentence, data.label, test_size=0.2, random_state=42)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train_word)
    
    max_length = 150
    vocab_size = len(tokenizer.word_index)+1
    X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_word), maxlen=max_length)
    X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_word), maxlen=max_length)
    print(f'shape of training data: {X_train.shape}')
    print(f'shape of test data: {X_test.shape}')
    print(f'maximum index: {vocab_size}')
    print(f'maximum index of training data: {np.max(X_train)}')


    # Word to Vector
    embedding_dim = 250
    w2v_model = Word2Vec(X_train_word, min_count=1, vector_size=embedding_dim, epochs=10, sg=1)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    
    print(f'shape of embedding_matrix: {embedding_matrix.shape}')


    # Training set up
    
    ## CNN Model
    cnn_model = Sequential()

    embedding_layer = Embedding(
        vocab_size, embedding_dim,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=False)

    cnn_model.add(embedding_layer)
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Conv1D(128, 16, activation='relu'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv1D(128, 16, activation='relu'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv1D(128, 16, activation='relu'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(2, activation='softmax'))
    cnn_model.summary()

    callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    cnn_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    cnn_record = cnn_model.fit(
        X_train, y_train,
        batch_size=1024, epochs=30,
        validation_split=0.2,
        callbacks=[callback],
        verbose=1)

    test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=2)
    print(f'CNN accuracy: {test_acc}')
    plot_acc(cnn_record)
    plot_loss(cnn_record)


    ## LSTM Model
    lstm_model = Sequential()

    embedding_layer = Embedding(
        vocab_size, embedding_dim,
        weights=[embedding_matrix],
        input_length=max_length, trainable=False)

    lstm_model.add(embedding_layer)
    lstm_model.add(Dropout(0.5))
    lstm_model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.summary()

    lstm_model.compile(
        loss='binary_crossentropy',
        optimizer="adam",
        metrics=['accuracy'])

    lstm_record = lstm_model.fit(
        X_train, y_train,
        batch_size=1024, epochs=30,
        validation_split=0.2, verbose=1)

    test_loss, test_acc = lstm_model.evaluate(X_test, y_test, verbose=2)
    print(f'LSTM accuracy: {test_acc}')
    plot_acc(lstm_record)
    plot_loss(lstm_record)
