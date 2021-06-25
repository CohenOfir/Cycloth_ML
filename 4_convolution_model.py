from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

train_path = '/path/to/train/dir'
valid_path = '/path/to/validation/dir'
test_path = '/path/to/test/dir'

labels = ['Coat', 'Dress', 'Pants', 'Shirt', 'Shoes', 'Skirt', 'Top']

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), classes=labels,
                                                         batch_size=16)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), classes=labels,
                                                         batch_size=4)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, validation_data=valid_batches, epochs=10, verbose=1)

model.save("4_convolution_model.h5")
