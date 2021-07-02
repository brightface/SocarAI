import tensorflow as tf
import pandas as pd
import numpy as np
from itertools import chain
EPOCHS = 1000
NUM_WORDS = -10000
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
import torch


def open_file(path, encode):
    csv = pd.read_csv(path, encoding = encode)
    csv = csv.values
    res = []

    for i in csv:
        res.append(i)

    return res

class MyModel(tf.keras.Model):
   
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 16)  # 길이가 10000인 16으로 바꿔준다. float으로 바꾸기 위해.
        self.rnn = tf.keras.layers.LSTM(32)
        # self.tmp = tf.keras.layers.Bidirectional(
            # self.rnn, merge_mode='concat', weights=None, backward_layer=None
            # )
        self.dense = tf.keras.layers.Dense(3, activation='softmax')  # 감정분석, 길이가 2인 softmax 함수

    def call(self, x, training=None, mask=None):
        x = self.emb(x)
        x = self.rnn(x)
        # x = self.tmp(x)
        return self.dense(x)

# Implement training loop
@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images, training=False)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


##main

# imdb = tf.keras.datasets.imdb#imdb에서 가져온다.

# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS) #단어를 몇개로 할것인가. 정해준다.
x_train_csv = open_file("./input_files/xx_acc.csv","UTF-8") + open_file("./input_files/xx_non_acc.csv","UTF-8")
y_train_csv = open_file("./input_files/yy_acc.csv","UTF-8") + open_file("./input_files/yy_non_acc.csv","UTF-8")
x_train_csv = np.array(x_train_csv)
y_train_csv = np.array(y_train_csv)

x_train, x_test, y_train, y_test  = train_test_split(x_train_csv,y_train_csv,test_size=0.2, random_state= 666)

print(x_train.shape)

print(y_train.shape)
    #x_train 길이가 x_test와 다를수 있다 그래서 특정길이로 잘라줘야한다. 그길이에 미치지 못하는것들은 패딩도해줘야한다.
#앞으로 패딩을 해줄것이다.
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                        dtype=float,
                                                       value=0,
                                                       padding='pre',
                                                       maxlen=120)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                       dtype=float,
                                                      value=0,
                                                      padding='pre',
                                                      maxlen=120)

# y_train = tf.keras.preprocessing.sequence.pad_sequences(y_train,
#                                                         dtype=float,
#                                                        value=0,
#                                                        padding='pre',
#                                                        maxlen=None)

# y_test = tf.keras.preprocessing.sequence.pad_sequences(y_test,
#                                                        dtype=float,
#                                                       value=0,
#                                                       padding='pre',
#                                                       maxlen=None)

print(x_train.shape)
print(y_train.shape)
#tensor_slices(x_train, y_train)을 가져올수 있따. 그리고 shuffle하고 배치크기 32
x_train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
x_test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32) #test도 그대로 한다.

# print(train_ds)
# print(test_ds)
# Create model
model = MyModel()
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(3, return_sequences=True), input_shape=(32, 3)))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(3)))
# model.add(tf.keras.layers.Dense(32))
# model.add(tf.keras.layers.Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#  # With custom backward layer
# model = tf.keras.Sequential()
# forward_layer = tf.keras.layers.LSTM(3, return_sequences=True)
# backward_layer = tf.keras.layers.LSTM(3, activation='relu', return_sequences=True,
#                       go_backwards=True)
# model.add(tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer,
#                         input_shape=(32, 3)))
# model.add(tf.keras.layers.Dense(32))
# model.add(tf.keras.layers.Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
  
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.)

# print(x_test.shape)
# print(x_train.shape)
# print(y_train.shape)
# print(y_test.shape)
# Define loss and optimizer
# loss_object = tf.keras.losses.sparse_categorical_crossentropy()
loss_object = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy()
# print(x_train)
for epoch in range(EPOCHS):
    for seqs, labels in x_train_ds:
        train_step(model, seqs, labels, loss_object, optimizer, train_loss, train_accuracy)
        # print(train_step)

    for test_seqs, test_labels in x_test_ds:
        test_step(model, test_seqs, test_labels, loss_object, test_loss, test_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
