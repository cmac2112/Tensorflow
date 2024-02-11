#learning through tutorial on tensorflow website, 

#import what we need
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#program will be able to determine type of clothing
#using fashion mnist dataset

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#images are 28x28 numpy arrays, pixel values range from 0 to 255
#similar to observatory camera, could be used in a personal project there sometime

#each image is mapped to a single label, since the class names are not included
#store them here to use later when plotting images

class_names = ['T-shirt/top', 'Pants', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Shoe']

y = train_images.shape
x = len(train_labels)
z = train_labels
a = test_images.shape
print(x, y , z, a)

#preprocess the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
#plt.show()

#scale the values to a range of 0 to 1 before feeding to the neural network model
train_images = train_images / 255.0
test_images = test_images / 255.0

#verify data is corret, display first 25 images from training set
plt.figure(figsize=(10,10))
#for i in range(25):
    #plt.subplot(5,5,i+1)
    #plt.xticks([])
    #plt.yticks([])
    #plt.grid(False)
    #plt.imshow(train_images[i], cmap=plt.cm.binary)
    #plt.xlabel(class_names[train_labels[i]])
#plt.show()

#build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #28x28 because its a 28x28 image
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense(10)
])
#Flatten transforms the format of the images from a 2d array to a 1d array
#of 28*28=784 pixels

#compile the model

model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

#train the model
#1. feed data into the model, training data is train_images and train_labels
#2. model learns to associate images and labels
#3. ask model to make predictions about a test set, ex test_images
#4. verify predictions match labels from test labels

model.fit(train_images, train_labels, epochs=50)

#evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#make predictions
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
testing = predictions[0]
print(testing)
highest = np.argmax(predictions[0])
print(highest)
#prints out the value corresponding to the class label from before
#should be 9 which corresponds to ankle boot

#define function to plot full set of 10 class predictions

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
    
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#verify predictions
    #with model trained, we can use it to make predictions about some images
    #looking at 0th image, correct predictions are blue
    #number gives percentage for the predicted label
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

#plotting several images with their predictions
#model can be wrong even when confident

num_rows = 5
num_col = 3
num_images = num_rows*num_col
plt.figure(figsize=(2*2*num_col, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_col, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_col, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#use trained model to make a prediction about a single image

img = test_images[1]
print(img.shape)

#add the image to a batch where its the only memeber
img = (np.expand_dims(img, 0))
print(img.shape)

predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
print(np.argmax(predictions_single[0]))