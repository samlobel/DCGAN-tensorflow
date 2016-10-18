from __future__ import print_function
from __future__ import absolute_import
from loader import Loader

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
print('loader')


# def load_mnist():
#     data_dir = os.path.join("./data", self.dataset_name)
    
#     fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
#     loaded = np.fromfile(file=fd,dtype=np.uint8)
#     trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

#     fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
#     loaded = np.fromfile(file=fd,dtype=np.uint8)
#     trY = loaded[8:].reshape((60000)).astype(np.float)

#     fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
#     loaded = np.fromfile(file=fd,dtype=np.uint8)
#     teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

#     fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
#     loaded = np.fromfile(file=fd,dtype=np.uint8)
#     teY = loaded[8:].reshape((10000)).astype(np.float)

#     trY = np.asarray(trY)
#     teY = np.asarray(teY)
    
#     X = np.concatenate((trX, teX), axis=0)
#     y = np.concatenate((trY, teY), axis=0)
    
#     seed = 547
#     np.random.seed(seed)
#     np.random.shuffle(X)
#     np.random.seed(seed)
#     np.random.shuffle(y)
    
#     y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
#     for i, label in enumerate(y):
#         y_vec[i,y[i]] = 1.0
    
#     return X/255.,y_vec

# Maybe I should make some not of size.
class MNIST_Loader(Loader):
  def __init__(self, batch_size):
    self.batch_size = batch_size
    # X, _ = load_mnist()
    self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # self.X = X
    # if (len(X) <= batch_size):
    #   raise Exception("Batch must be smaller than total!")
    # counter = 0

  def get_config(self):
    config = {
      'batch_size' : self.batch_size,
      'image_w' : 28,
      'image_h' : 28,
      'image_z' : 1
    }
    return config

  def retrieve(self):
    # Only 50,000. But that really shouldn't be such an issue.
    batch_xs, _ = mnist.train.next_batch(self.batch_size)
    return batch_xs
    # X = self.X
    # if counter + self.batch_size >= len(X): #IS GTE OR JUST GT?
    #   to_return = np.concatenate(X[counter:], x[0:counter + batch_size - len(X)])
    # else:
    #   to_return = X[counter:counter+batch_size]
    # if len(to_return) != self.batch_size:
    #   print('size of to return is {} but should be {}'.format(len(to_return), batch_size))
    #   raise Exception("Wrong batch size returned")
    # counter = (counter + batch_size) % len(X)
    # return to_return

    








