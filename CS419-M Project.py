# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


#Parameters
learning_rate = 0.001
epochs = 150
bath_size = 633

#Neural Network Parameters
hidden_layer1 = 64 #1st layer no of neurons
hidden_layer2 = 256 #2nd layer no of neurons
hidden_layer3 = 256 #3rd layer no of neurons
hidden_layer4 = 64 #4th layer no of neurons
output_layer = 2 #No of neurons in output layer
input_no = 20 # 20 audio features extracted from a audio sample

#label has been converted to integer as 0 for male and 1 for female
male_class = 0 
female_class = 1

#Weights matrix
weights = {
    'Hl1': tf.Variable(tf.random_normal([input_no, hidden_layer1])),
    'Hl2': tf.Variable(tf.random_normal([hidden_layer1, hidden_layer2])),
    'Hl3': tf.Variable(tf.random_normal([hidden_layer2, hidden_layer3])),
    'Hl4': tf.Variable(tf.random_normal([hidden_layer3, hidden_layer4])),
    'O': tf.Variable(tf.random_normal([hidden_layer4, output_layer]))
}

#Biases matrix
biases = {
    'b1': tf.Variable(tf.random_normal([hidden_layer1])),
    'b2': tf.Variable(tf.random_normal([hidden_layer2])),
    'b3': tf.Variable(tf.random_normal([hidden_layer3])),
    'b4': tf.Variable(tf.random_normal([hidden_layer4])),
    'o': tf.Variable(tf.random_normal([output_layer]))
}

#Output layer

def MLP_output(x):
	layer_1 = tf.add(tf.matmul(x, weights['Hl1']), biases['b1'])
	layer_2 = tf.add(tf.matmul(layer_1,weights['Hl2']), biases['b2'])
	layer_3 = tf.add(tf.matmul(layer_2, weights['Hl3']), biases['b3'])
	layer_4 = tf.add(tf.matmul(layer_3, weights['Hl4']), biases['b4'])
	out = tf.add(tf.matmul(layer_4, weights['O']), biases['o'])
	return out

out = MLP_output(x)

