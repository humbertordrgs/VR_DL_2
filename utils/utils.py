import os
def load_images(image_gallery_path, mapping_file_path):
  class_map = {}
  with open(mapping_file_path,'r') as f1:
    content = f1.read().splitlines()
    for line in content:
      parts = line.split(sep="\t")
      class_map[parts[0].replace('-','_')] = parts[1]  
  images = {}
  if (os.path.isdir(image_gallery_path)):
    for class_name in os.listdir(image_gallery_path):
      class_idx = class_map[class_name]
      if (class_idx not in images):
        images[class_idx] = []
      for file_name in os.listdir(image_gallery_path + '/' + class_name):
        images[class_idx].append(file_name)
  return images