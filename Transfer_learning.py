from keras.applications.vgg16 import VGG16
from tensorflow.keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_path = '/path/to/train/dir'
valid_path = '/path/to/validation/dir'
test_path = '/path/to/test/dir'

labels = ['Coat', 'Dress', 'Pants', 'Shirt', 'Shoes', 'Skirt', 'Top']

# Load datasets
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), classes=labels,
                                                         batch_size=16)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), classes=labels,
                                                         batch_size=4)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), classes=labels,
                                                        batch_size=4)
# Create VGG16 model instance
vgg16_model = VGG16()

model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)

# Remove last Dense layer of VGG16
model.layers.pop()

# freeze layers
for layer in model.layers:
    layer.trainable = False

# Ad dense layer for the 7 categories
model.add(Dense(7, activation='softmax'))

model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, validation_data=valid_batches, epochs=10, verbose=1)

predictions = model.predict(x=test_batches)

model.save("Transfer_learning_model.h5")
