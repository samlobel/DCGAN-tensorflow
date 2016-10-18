import numpy as np
import tensorflow as tf


# Should have an init method, and then a retrieve method. I guess that's really
# it.
class Loader(object):
  def __init__(self, batch_size):
    raise NotImplementedError("init not implemented")

  def retrieve(self):
    raise NotImplementedError("Retrieve not implemented")

  def get_config(self):
    raise NotImplementedError("get_config not implemented")
