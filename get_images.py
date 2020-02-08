import matplotlib.pyplot as plt
from annoy import AnnoyIndex
import train
from train import image_size,n_channels, model_save_path, encoding_vector_length, annoy_file_name
from images import train_images

def get_similar_images(image, n_similar=10):
  encoding = train.get_encodings(model_save_path,image.reshape(-1, image_size, image_size,n_channels), encoding_vector_length)
  saved_ann = AnnoyIndex(encoding_vector_length)
  saved_ann.load(annoy_file_name)
  nn_indices = saved_ann.get_nns_by_vector(encoding[0], n_similar)
  print("Similar images are")
  for i, index in enumerate(nn_indices,1):
    image = train_images[index].reshape(28,28)
    plt.subplot(2,5,i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap='Greys_r')