import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


image = plt.imread('edge_detection_ex.jpg')
plt.imshow(image)
plt.show()
image = image.reshape((1,720,1280,3))
image = tf.constant(image,dtype=tf.float64)

weight = np.array([[[-1,-2,-1],[0,0,0],[1,2,1]],[[-1,-2,-1],[0,0,0],[1,2,1]],[[-1,-2,-1],[0,0,0],[1,2,1]]])
weight = weight.reshape((1,3,3,3))
weight_init = tf.constant_initializer(weight)

conv2d= keras.layers.Conv2D(filters=1, kernel_size=3,padding='SAME',kernel_initializer=weight_init)(image)
plt.imshow(conv2d.numpy().reshape(720,1280),cmap='gray')
plt.show()

