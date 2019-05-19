### import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile
import pathlib
import shutil
import random
import tensorflow as tf
import keras
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

###function to create folders
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

###function to move images        
def move(from_dir, to_dir):
    files = os.listdir(from_dir)
    split = int(0.9 * len(files))
    valid = files[split:]
    for f in valid:
        shutil.move(from_dir + "/"+f, to_dir)

###loading the data and extract it
local_zip = 'data.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()
data_root = pathlib.Path('train-data')
#print(data_root)

###print the folders paths (labels)
for item in data_root.iterdir():
  print(item)
  

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
image_count

### sort the labels alphbitically (as the arrangement of the labels in the model)
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((index,name) for index,name in enumerate(label_names))
label_to_index




### Directories with our training images
train_Baby_Bananas_dir = os.path.join('train-data/Baby Bananas')
train_Cavendish_Bananas_dir = os.path.join('train-data/Cavendish Bananas')
train_boat_dir = os.path.join('train-data/boat')
train_Gala_Apple_dir = os.path.join('train-data/Gala Apple')
train_Golden_Delicious_Apple_dir = os.path.join('train-data/Golden Delicious Apple')
train_brie_cheese_dir = os.path.join('train-data/brie cheese')
train_feta_cheese_dir = os.path.join('train-data/feta cheese')
train_Gouda_cheese_dir = os.path.join('train-data/Gouda cheese')
train_Parmigian_cheese_dir = os.path.join('train-data/Parmigian cheese')


###make validation directory and move (10%) of the train-data to it.
createFolder('./valid-data/')

createFolder('./valid-data/Baby Bananas') ; valid_Baby_Bananas= 'valid-data/Baby Bananas'
createFolder('./valid-data/boat') ; valid_boat= 'valid-data/boat'
createFolder('./valid-data/brie cheese') ; valid_brie_cheese= 'valid-data/brie cheese'
createFolder('./valid-data/Cavendish Bananas') ; valid_Cavendish_Bananas= 'valid-data/Cavendish Bananas'
createFolder('./valid-data/feta cheese') ; valid_feta_cheese= 'valid-data/feta cheese'
createFolder('./valid-data/Gala Apple') ; valid_Gala_Apple= 'valid-data/Gala Apple'
createFolder('./valid-data/Golden Delicious Apple'); valid_Golden_Delicious_Apple= 'valid-data/Golden Delicious Apple'
createFolder('./valid-data/Gouda cheese') ; valid_Gouda_cheese= 'valid-data/Gouda cheese'
createFolder('./valid-data/Parmigian cheese') ; valid_Parmigian_cheese= 'valid-data/Parmigian cheese'

move(train_Baby_Bananas_dir,valid_Baby_Bananas)
move(train_Cavendish_Bananas_dir,valid_Cavendish_Bananas)
move(train_boat_dir,valid_boat)
move(train_Gala_Apple_dir,valid_Gala_Apple)
move(train_Golden_Delicious_Apple_dir,valid_Golden_Delicious_Apple)
move(train_brie_cheese_dir,valid_brie_cheese)
move(train_feta_cheese_dir,valid_feta_cheese)
move(train_Gouda_cheese_dir,valid_Gouda_cheese)
move(train_Parmigian_cheese_dir,valid_Parmigian_cheese)

###ImageDataGenerator for training and validation
TRAINING_DIR = 'train-data'

###using rescale and augmentation on the fly
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


VALIDATION_DIR = "valid-data"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical'
)


###the convolutional neural network
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
	#to prevent overfitting
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # 9 output layer
    tf.keras.layers.Dense(9, activation='softmax')
])


model.summary()


model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit_generator(train_generator, epochs=100, validation_data = validation_generator, verbose = 1)

model.save("model.h5")

### plot the training acc with validation accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

###testing
path = os.listdir('test-data')
csv= []
for fn in path:
 
  # predicting images
  path1 = 'test-data/' + fn  
  img = image.load_img(path1, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  result = np.where(classes == np.amax(classes))
  csv.append(fn + ',' + label_to_index[int(result[1])])
  #print(fn, classes, result[1], label_to_index[int(result[1])])
 
###csv
output= pd.DataFrame(csv) 
output.columns = ['image_name,class']
output.to_csv('out.csv', header =True , index=None)