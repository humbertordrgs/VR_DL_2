import os
import time
from PIL import Image
from torchvision import transforms

preprocess_pipeline = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
])

def get_class_map(mapping_file_path):
  class_map = {}
  with open(mapping_file_path,'r') as f1:
    content = f1.read().splitlines()
    for line in content:
      parts = line.split(sep="\t")
      class_map[parts[1]] = parts[0].replace('-','_')
  return class_map

def load_images(image_gallery_path): 
  images = {}
  if (os.path.isdir(image_gallery_path)):
    for class_name in os.listdir(image_gallery_path):
      if (class_name not in images):
        images[class_name] = []
      for file_name in os.listdir(image_gallery_path + '/' + class_name):
        images[class_name].append(file_name)
  return images

def get_processed_img(img_path):
  image = Image.open(img_path)
  return preprocess_pipeline(image)

def execution_time(func):
  def exec(*args, **kwargs):
    start_time = time.time()
    res = func(*args, **kwargs)
    print(
      "{function_name}: {time:.5f} seconds".format(
        function_name= func._name_ if func.getattr('__name__', None) else 'Constructor',
        time=(
          time.time() - start_time
        )
      )
    )
    return res
  return exec