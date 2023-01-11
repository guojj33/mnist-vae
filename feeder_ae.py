from torchvision.datasets import MNIST
import numpy as np

mnist_root = './mnist_data'

class MNIST_Feeder():
  def __init__(self, phase) -> None:      
    self.dataset = MNIST(root='./mnist_data', train=(phase=='train'))

  def __getitem__(self, idx):
    x, y = self.dataset[idx]
    x = np.array(x) / 255 # [28, 28] 归一化
    x = x[None, :, :] # [1, 28, 28]
    return x, x.copy(), y
  
  def __len__(self):
    return len(self.dataset)