from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# Load model from file
model = load_model('model_name')

# Define labels
labels = ['Coat', 'Dress', 'Pants', 'Shirt', 'Shoes', 'Skirt', 'Top']

# Load image for prediction
image = load_img('image_path', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)

# Get top 3 classes
print(yhat[0])
top1 = labels[yhat.argmax()]
yhat[0][yhat.argmax()] = -1
top2 = labels[yhat.argmax()]
yhat[0][yhat.argmax()] = -1
top3 = labels[yhat.argmax()]

print(top1, top2, top3)