import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from feeder_ae import MNIST_Feeder
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from vae import Classifier
import torch.nn.functional as F

def to_image(x_re):
  x_re *= 255
  x_re = x_re.clip(0, 255)
  x_re = x_re.astype(np.int8)
  # x_re = rearrange(x_re, 'C H W -> H W C')
  x_re = Image.fromarray(x_re)
  return x_re

device = torch.device('cuda:0')

workdir = './workdir'
import os
dirs = os.listdir(workdir)
for i, d in enumerate(dirs):
  print('[{}] {}'.format(i, d))
idx = int(input('index of dir to load:'))
workdir = '{}/{}'.format(workdir, dirs[idx])
print('loading models from {} ...'.format(workdir))
# workdir = './results_dim_20'
# workdir = './workdir/2023-01-11_15-20-21'

encoder = torch.load('{}/encoder.pth.tar'.format(workdir))
decoder = torch.load('{}/decoder.pth.tar'.format(workdir))
dim_z = decoder.dim_z
n_hidden = decoder.n_hidden
classifier = torch.load('{}/classifier.pth.tar'.format(workdir))

for p in encoder.parameters():
    p.requires_grad = False
for p in decoder.parameters():
    p.requries_grad = False
for p in classifier.parameters():
    p.requires_grad = False

def generateImage(target_label):
  zs_target = torch.randn([1, dim_z]).to(device)
  zs_target.requires_grad = True
  optimizer = Adam([zs_target], lr=0.001)
  loss_class = nn.CrossEntropyLoss()
  ys_target = torch.tensor([target_label]).to(device)

  i = 1
  while True:
    optimizer.zero_grad()
    xs_target_re = decoder(zs_target)
    hiddens_target = decoder.getHidden(zs_target)
    logits_target = classifier(hiddens_target)
    loss = loss_class(logits_target, ys_target)
    loss.backward()
    optimizer.step()
    probs = F.softmax(logits_target, dim=1)
    print('iteration: {}, loss: {}, prob: {}'.format(i, loss.item(), probs[0]))
    label = logits_target.max(1)[1].item()
    # if label == target_label:
    #   break
    if probs[0][target_label] > 0.8:
      break
  xs_target_re = xs_target_re.reshape(1, 1, 28, 28)
  x_target_re = xs_target_re[0].detach().squeeze().cpu().numpy()
  image_target = to_image(x_target_re)
  plt.imshow(image_target)
  plt.show()

def generateImage2(target_label):
  while True:
    zs_target = torch.randn([1, dim_z]).to(device)
    xs_target_re = decoder(zs_target)
    hiddens_target = decoder.getHidden(zs_target)
    logits_target = classifier(hiddens_target)
    probs = F.softmax(logits_target, dim=1)
    label = logits_target.max(1)[1].item()
    if probs[0][target_label] > 0.95:
      break
  xs_target_re = xs_target_re.reshape(1, 1, 28, 28)
  x_target_re = xs_target_re[0].detach().squeeze().cpu().numpy()
  image_target = to_image(x_target_re)
  plt.imshow(image_target)
  plt.show() 

def classifyImage():
  zs_target = torch.randn([1, dim_z]).to(device)

  xs_target_re = decoder(zs_target)
  hiddens_target = decoder.getHidden(zs_target)
  logits_target = classifier(hiddens_target)
  probs = F.softmax(logits_target, dim=1)
  label = logits_target.max(1)[1].item()

  xs_target_re = xs_target_re.reshape(1, 1, 28, 28)
  x_target_re = xs_target_re[0].detach().squeeze().cpu().numpy()
  image_target = to_image(x_target_re)
  plt.imshow(image_target)
  plt.title('label: {}'.format(label))
  plt.show()