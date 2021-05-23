import torch.nn as nn
import torch.nn.functional as F
class Conv1x1(nn.Module):
  def __init__(self, **kwargs):
    super(Conv1x1, self).__init__()
    self.conv = nn.Conv2d(
      kernel_size=1,
      **kwargs
    )
  def forward(self, input):
    return self.conv(input)

class Conv3x3(nn.Module):
  def __init__(self, **kwargs):
    super(Conv3x3, self).__init__()
    self.conv = nn.Conv2d(
      kernel_size=3,
      **kwargs
    )
  def forward(self, input):
    return self.conv(input)

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, increase_initial_stride=False):
    super(ResidualBlock, self).__init__()

    self.seq_1 = nn.Sequential(
      # Pre activation
      nn.BatchNorm2d(in_channels),
      nn.ReLU(),
      Conv3x3(
        padding=1,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=2 if increase_initial_stride else 1,
        bias=False
      ),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      Conv3x3(
        in_channels=out_channels,
        out_channels=out_channels,
        stride=1,
        padding=1,
        bias=False
      )
    )
    
    self.increase_initial_stride = increase_initial_stride
    self.in_ch = in_channels
    self.out_ch = out_channels
  
    self.conv_aux = Conv1x1(
      in_channels=in_channels,
      out_channels=out_channels,
      stride=2 if increase_initial_stride else 1,
      bias=False
    )

  def forward(self, x):

    partial_res = self.seq_1(x)

    # Using additional 1x1 conv layer to be able to add the residual
    # In the specific needed cases
    if self.in_ch != self.out_ch or self.increase_initial_stride:
      x = self.conv_aux(x)

    # Applying residual connection
    partial_res = partial_res + x
    
    return partial_res

class ResNet34Backbone(nn.Module):
  def __init__(self):
    super(ResNet34Backbone, self).__init__()
    self.network_sequence = nn.Sequential(
      nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False),
      nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
      
      # RGroup 1
      ResidualBlock(64,64,3),
      ResidualBlock(64,64,3),
      ResidualBlock(64,64,3),

     # RGroup 2
      ResidualBlock(64,128,3,increase_initial_stride=True),
      ResidualBlock(128,128,3),
      ResidualBlock(128,128,3),
      ResidualBlock(128,128,3),

      # RGroup 3
      ResidualBlock(128,256,3,increase_initial_stride=True),
      ResidualBlock(256,256,3),
      ResidualBlock(256,256,3),
      ResidualBlock(256,256,3),
      ResidualBlock(256,256,3),
      ResidualBlock(256,256,3),

      # RGroup 4
      ResidualBlock(256,512,3,increase_initial_stride=True),
      ResidualBlock(512,512,3),
      ResidualBlock(512,512,3),

      # activation of the last block
      nn.BatchNorm2d(512),
      nn.ReLU()
    )
  def forward(self, x):
    return self.network_sequence(x)

class SiameseNet(nn.Module):
  def __init__(self, sketch_classes, use_triplets=False):
    super(SiameseNet, self).__init__()
    self.use_triplets = use_triplets
    self.sketch_backbone = ResNet34Backbone()
    self.image_net_backbone = ResNet34Backbone()

    self.common_seq = nn.Sequential(
      nn.AvgPool2d(kernel_size=7,stride=1),
      nn.Flatten(),
      nn.Linear(512,512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Linear(512,512)
    )

    self.classifier = nn.Linear(512,sketch_classes)

  
  def forward(self, input):
    
    ###################### SKETCHES ############################
    res_sketch = self.common_seq(self.sketch_backbone(input[0]))
    sketch_logits = self.classifier(res_sketch)
    sketch_embeddings = F.normalize(res_sketch)

    ##################### POSITIVES ############################
    res_pos = self.common_seq(self.image_net_backbone(input[1]))
    postive_logits = self.classifier(res_pos)
    postive_embeddings = F.normalize(res_pos)

    embeddings = [sketch_embeddings, postive_embeddings]

    logits = [sketch_logits, postive_logits]

    ##################### NEGATIVES ############################
    if self.use_triplets:
      res_neg = self.common_seq(self.image_net_backbone(input[2]))
      negative_logits = self.classifier(res_neg)
      negative_embeddings = F.normalize(res_neg)
      embeddings.append(negative_embeddings)
      logits.append(negative_logits)

    return embeddings, logits