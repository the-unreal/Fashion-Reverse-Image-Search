import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# resize the pixel values between 0 and 255
train_images = train_images/255
test_images = test_images/255