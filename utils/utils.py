import os
def load_images(path):
  images = {}
  if (os.path.isdir(path)):
    for class_name in os.listdir(path):
      if (class_name not in images):
        images[class_name] = []
      for file_name in os.listdir(path + '/' + class_name):
        images[class_name].append(file_name)
  return images