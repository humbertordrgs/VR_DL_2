# PyTorch and numpy related
from torch.utils.data import Dataset

# External tool (built by us) to format the dataset structure
from utils import load_images, get_processed_img, execution_time

class SBIRImageGalleryTestDataset(Dataset):
  
  @execution_time
  def __init__(self, image_gallery_folder_path):
      
    self.images = []
    structured_images = load_images(image_gallery_folder_path)
    for key in structured_images:
      for images in structured_images[key]:  
        image_path = f"{image_gallery_folder_path}/{key}/{images}"
        self.images.append(get_processed_img(image_path))

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    return self.images[idx]
