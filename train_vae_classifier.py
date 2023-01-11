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

def to_image(x_re):
  x_re *= 255
  x_re = x_re.clip(0, 255)
  x_re = x_re.astype(np.int8)
  # x_re = rearrange(x_re, 'C H W -> H W C')
  x_re = Image.fromarray(x_re)
  return x_re

device = torch.device('cuda:0')

# workdir = './results_dim_20'
workdir = './workdir/2023-01-11_15-20-21'
encoder = torch.load('{}/encoder.pth.tar'.format(workdir))
decoder = torch.load('{}/decoder.pth.tar'.format(workdir))
dim_z = decoder.dim_z
n_hidden = decoder.n_hidden
for p in encoder.parameters():
    p.requires_grad = False
for p in decoder.parameters():
    p.requries_grad = False

classifier = Classifier(n_hidden).to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = Adam(classifier.parameters(), lr=0.001)

train_batch_size = 4096
eval_batch_size = 4096
train_feeder = MNIST_Feeder(phase='train')
test_feeder = MNIST_Feeder(phase='test')
train_dataloader = DataLoader(train_feeder, batch_size=train_batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_feeder, batch_size=eval_batch_size, shuffle=False, drop_last=False)

def get_sample():
    zs = torch.randn([1, decoder.dim_z]).to(device)
    rec = decoder(zs).reshape(-1, 28, 28).detach().cpu().numpy()[0]
    image = to_image(rec)
    plt.imshow(image)
    plt.show()
# summary(encoder, (1, 1*28*28))
# dim_z = 20
# summary(decoder, (1, dim_z))

epoch = 60
for e in range(epoch):
  print("EPOCH {}/{}".format(e, epoch))
  # train
  encoder.train()
  decoder.train()
  classifier.train()
  optimizer.zero_grad()

  train_iter = tqdm(train_dataloader, dynamic_ncols=True)
  num_sample = 0
  num_top1 = 0
  train_loss = []
  for idx, (x, _, y) in enumerate(train_iter):
    x = x.float().to(device)
    x = x.reshape(x.shape[0], -1)
    y = y.long().to(device)
    mean, stddev = encoder(x)
    z = mean + torch.randn_like(mean)*stddev
    hidden = decoder.getHidden(z)

    logits = classifier(hidden)
    predictions = logits.max(1)[1]
    loss = loss_func(logits, y)

    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())
    num_top1 += predictions.eq(y).sum().item()
    num_sample += x.shape[0]

  top1_acc = num_top1 / num_sample
  train_loss = np.array(train_loss)
  train_loss = train_loss.sum() / len(train_loss)
  print('train loss: {}, acc: {}'.format(train_loss, top1_acc))

  # test
  eval_iter = tqdm(test_dataloader, dynamic_ncols=True)
  with torch.no_grad():
    encoder.eval()
    decoder.eval()
    classifier.eval()
    eval_loss = []
    num_top1 = 0
    num_sample = 0
    for idx, (x, _, y) in enumerate(eval_iter):
      x = x.float().to(device)
      x = x.reshape(x.shape[0], -1)
      y = y.long().to(device)
      mean, stddev = encoder(x)
      z = mean + torch.randn_like(mean)*stddev

      hidden = decoder.getHidden(z)
      logits = classifier(hidden)
      predictions = logits.max(1)[1]

      loss = loss_func(logits, y)

      eval_loss.append(loss.item())
      num_top1 += predictions.eq(y).sum().item()
      num_sample += x.shape[0]
    top1_acc = num_top1 / num_sample

    eval_loss = np.array(eval_loss)
    eval_loss = eval_loss.sum() / len(eval_loss)
    print('eval loss: {}, acc: {}'.format(eval_loss, top1_acc))
torch.save(classifier, './classifier.pth.tar')