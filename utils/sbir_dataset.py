# Generic python libraries
import random
from random import sample

# PyTorch and numpy related
from torch.utils.data import Dataset

# External tool (built by us) to format the dataset structure
from utils import load_images, get_class_map, get_processed_img, execution_time

class SketchBasedImageRetrievalDataset(Dataset):
  
  @execution_time
  def __init__(
    self, sketch_folder_path, sketch_index_file, \
    image_gallery_folder_path, mapping_file_path, use_triplets = False \
  ):

    self.use_triplets = use_triplets
    self.sketches = []
    self.classes = []
    self.positive_images = []
    if use_triplets:
      self.negative_images = []

    class_map = get_class_map(sketch_folder_path + "/" + mapping_file_path)
    structured_images = load_images(image_gallery_folder_path)
    index_file_path = sketch_folder_path + "/" + sketch_index_file

    with open(index_file_path, "r") as sketch_file:
      # Proceded to reduce the size of the source datasets to 8000 due to hardware limitations
      # sketch_lines = sketch_file.readlines()
      sketch_lines = random.sample(sketch_file.readlines(), 8000)
      for line in sketch_lines:
        sketch_path, sketch_idx = line.split()
        sketch_path = sketch_folder_path + "/" + sketch_path
        sketch_class = class_map[sketch_idx]

        positive_random_image = sample(structured_images[sketch_class], 1)[0]
        positive_image_path = f"{image_gallery_folder_path}/{sketch_class}/{positive_random_image}"

        self.sketches.append(get_processed_img(sketch_path))
        self.positive_images.append(get_processed_img(positive_image_path))
        self.classes.append(int(sketch_idx))

        if use_triplets:
          while True:
            negative_random_class_idx = str(random.randint(0, 249))
            if negative_random_class_idx != sketch_idx:
              negative_class = class_map[negative_random_class_idx]
              negative_random_image = sample(structured_images[negative_class], 1)[0]
              negative_image_path = f"{image_gallery_folder_path}/{negative_class}/{negative_random_image}"
              self.negative_images.append(get_processed_img(negative_image_path))
              break
  def __len__(self):
    return len(self.sketches)

  def __getitem__(self, idx):
    if self.use_triplets:
      return (self.sketches[idx], self.positive_images[idx], self.negative_images[idx]), self.classes[idx]
    return (self.sketches[idx], self.positive_images[idx]), self.classes[idx]