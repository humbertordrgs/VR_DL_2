import os

# PyTorch and numpy related
from torch.utils.data import Dataset

# External tool (built by us) to format the dataset structure
from utils import get_processed_img, execution_time

class SBIRTestSketchDataset(Dataset):
  
  @execution_time
  def __init__(self, sketch_folder_path):
      
    self.sketches = []

    for sketch_path in os.listdir(sketch_folder_path): 
        sketch_path = f"{sketch_folder_path}/{sketch_path}"
        self.sketches.append(get_processed_img(sketch_path))

  def __len__(self):
    return len(self.sketches)

  def __getitem__(self, idx):
    return self.sketches[idx]
