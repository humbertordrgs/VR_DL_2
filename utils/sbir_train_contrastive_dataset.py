# Generic python libraries
import random
from random import sample

# PyTorch and numpy related
from torch.utils.data import Dataset

# External tool (built by us) to format the dataset structure
from utils import load_images, get_class_map, get_processed_img, execution_time

class SBIRTrainContrastiveDataset(Dataset):
  
  @execution_time
  def __init__(
    self, sketch_folder_path, sketch_index_file, \
    image_gallery_folder_path, mapping_file_path, sub_sample=None \
  ):

    self.sketches = []
    self.images = []
    self.sketch_class = []
    self.image_class = []
    self.random_flag = []

    class_map = get_class_map(sketch_folder_path + "/" + mapping_file_path)
    structured_images = load_images(image_gallery_folder_path)
    index_file_path = sketch_folder_path + "/" + sketch_index_file

    with open(index_file_path, "r") as sketch_file:
      # Proceded to reduce the size of the source datasets to sub_sample due to hardware limitations
      if sub_sample is not None:
        sketch_lines = random.sample(sketch_file.readlines(), sub_sample)
      else:
        sketch_lines = sketch_file.readlines()

      for line in sketch_lines:
        sketch_path, sketch_idx = line.split()
        sketch_path = sketch_folder_path + "/" + sketch_path
        sketch_class = class_map[sketch_idx]

        self.sketches.append(get_processed_img(sketch_path))
        self.sketch_class.append(int(sketch_idx))

        random_flag = random.random() <= 0.5 
        if random_flag:
            random_image = sample(structured_images[sketch_class], 1)[0]
            image_path = f"{image_gallery_folder_path}/{sketch_class}/{random_image}"
            
            self.images.append(get_processed_img(image_path))
            self.image_class.append(int(sketch_idx))
            self.random_flag.append(1)
        else:
            while True:
                random_class_idx = str(random.randint(0, 249))
                if random_class_idx != sketch_idx:

                    random_class = class_map[random_class_idx]
                    random_image = sample(structured_images[random_class], 1)[0]
                    image_path = f"{image_gallery_folder_path}/{random_class}/{random_image}"
                    
                    self.images.append(get_processed_img(image_path))
                    self.image_class.append(int(random_class_idx))
                    self.random_flag.append(0)
                    break

  def __len__(self):
    return len(self.sketches)

  def __getitem__(self, idx):
    return (self.sketches[idx], self.images[idx]), (self.sketch_class[idx], self.image_class[idx], self.random_flag[idx])
