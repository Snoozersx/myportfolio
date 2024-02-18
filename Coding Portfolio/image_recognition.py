import tensorflow as tf
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from PIL import Image
import numpy as np

def load_data():
    (train_X, train_Y), (test_X, test_Y) = cifar10.load_data()
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = train_X / 255.0
    test_X = test_X / 255.0
    train_Y = np_utils.to_categorical(train_Y)
    test_Y = np_utils.to_categorical(test_Y)
    global num_classes
    num_classes = 10
    return train_X, train_Y, test_X, test_Y

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    sgd = SGD(lr=0.01, momentum=0.9, decay=(0.01 / 25), nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def plot_images(train_X, num_images_to_plot):
    fig, axes = plt.subplots(num_images_to_plot, 1, figsize=(20, 10 * num_images_to_plot))
    for i, ax in enumerate(axes):
        ax.imshow(train_X[i])
    plt.show()

def plot_predicted_images(train_X, predicted_classes, num_images_to_plot):
    fig, axes = plt.subplots(num_images_to_plot, 2, figsize=(20, 10 * num_images_to_plot))
    for i, ax in enumerate(axes):
        ax[0].imshow(train_X[i])
        ax[1].bar(range(num_classes), predicted_classes[i])
    plt.show()
   
class ImageRecognitionModel:
    def __init__(self):
        self.model = None

    def build_model(self):
        self.model = create_model()

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, train_X, train_Y):
        self.model.fit(train_X, train_Y, batch_size=16, epochs=10, verbose=1, validation_split=0.2)

    def evaluate_model(self, test_X, test_Y):
        _, accuracy = self.model.evaluate(test_X, test_Y)
        return accuracy

    def predict(self, image_path):
        img = Image.open(image_path)
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=3)
        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction)
        return predicted_class
    
def main():
    train_X, train_Y, test_X, test_Y = load_data()

    # Plot some images from the training set
    plot_images(train_X, 6)

    # Create the image recognition model
    model = ImageRecognitionModel()

    # Build the model
    model.build_model()

    # Compile the model
    model.compile_model()

    # Train the model
    model.train_model(train_X, train_Y)

    # Evaluate the model
    accuracy = model.evaluate_model(test_X, test_Y)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    # Predict the class of an example image
    image_path = "example_image.jpg"
    predicted_class = model.predict(image_path)
    print("Predicted class: ", predicted_class)

# Run the main function
if __name__ == "__main__":
    main()