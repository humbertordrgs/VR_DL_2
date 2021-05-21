import os
from PIL import Image

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

def get_processed_img(img_path, preprocess_pipeline):
  image = Image.open(img_path)
  return preprocess_pipeline(image)
