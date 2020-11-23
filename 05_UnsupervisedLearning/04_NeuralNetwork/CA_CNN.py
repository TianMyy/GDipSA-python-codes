import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot
from PIL import Image
import os
import glob
from sklearn.preprocessing import LabelEncoder




#set the dimension of the images
image1 = 80
image2 = 60

# label classification function
def classifyImage(name):
    if name[0] == "a":
        return "apple"
    elif name[0] == "b":
        return "banana"
    elif name[0] == "o":
        return "orange"
    elif name[0] == "m":
        return "mixed"
    else:
        raise Exception()

def render_digit(data):
    # note that matplotlib wants the format 
    # to be (width, height, number_of_channels)
    plot.imshow(data, cmap='gray')
    plot.show()


def preprocess(x_train, y_train, x_test, y_test):

    x_train = x_train / 255
    x_test = x_test / 255
    
    # Encode label
    le = LabelEncoder()
    le.fit(_y_train)
    y_train = le.transform(_y_train)
    y_test = le.transform(_y_test)

    # convert 1 dimensional array to 10-dimensional array
    # each row in y_train and y_test is one-hot encoded
    y_train = tf.keras.utils.to_categorical(y_train, 4)
    y_test = tf.keras.utils.to_categorical(y_test, 4)

    return (x_train, y_train, x_test, y_test)


def run_cnn(x_train, y_train, x_test, y_test):
    epoch = 20
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image1, image2, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Flatten())    
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    model = model.fit(x_train, y_train, batch_size=32, epochs=epoch, verbose=1,
        validation_data=(x_test, y_test))
        

    score = model.model.evaluate(x_test, y_test)
    print("score =", score)
    
    # visulazition of the plot 
    acc = model.history['accuracy']
    val_acc = model.history['val_accuracy']

    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochs_range = range(epoch)
    
    #training and validation accuracy plot
    plot.figure(figsize=(16, 8))
    plot.subplot(1, 2, 1)
    plot.plot(epochs_range, acc, label='Training Accuracy')
    plot.plot(epochs_range, val_acc, label='Validation Accuracy')
    plot.xlabel('number of epoch')
    plot.ylabel('accuracy score')
    plot.legend(loc='lower right')
    plot.title('Training and Validation Accuracy')
    
    #training and validation loss plot
    plot.subplot(1, 2, 2)
    plot.plot(epochs_range, loss, label='Training Loss')
    plot.plot(epochs_range, val_loss, label='Validation Loss')
    plot.xlabel('number of epoch')
    plot.ylabel('loss')
    plot.legend(loc='upper right')
    plot.title('Training and Validation Loss')
    plot.show()

    return

# main coding
print("Loading fruit image from dataset")

# read fruit image dataset
_x_train = []
_y_train = []
_x_test = []
_y_test = []

for f in glob.iglob("D:/curriculum/*.jpg"):
    im = Image.open(f).convert('RGB')
    newsize = (image1,image2)
    im = im.resize(newsize)
    _x_train.append(np.asarray(im))
    _y_train.append(classifyImage(os.path.basename(f)))

for f in glob.iglob("D:/curriculum/*.jpg"):
    im = Image.open(f).convert('RGB')
    newsize = (image1,image2)
    im = im.resize(newsize)
    _x_test.append(np.asarray(im))
    _y_test.append(classifyImage(os.path.basename(f)))

#transform 'list' to 'array' dtype 
_x_train = np.asarray(_x_train)
_y_train = np.asarray(_y_train)
_x_test = np.asarray(_x_test)
_y_test = np.asarray(_y_test)


print('x_train.shape = ',_x_train.shape)
print('x_test.shape = ',_x_test.shape)
print('y_test.shape = ',_y_test.shape)
print('y_train.shape = ',_y_train.shape)

(_x_train, _y_train, _x_test, _y_test) = preprocess(_x_train, _y_train,_x_test, _y_test)

model = run_cnn(_x_train, _y_train, _x_test, _y_test)



