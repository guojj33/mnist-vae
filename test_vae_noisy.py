import torch, os
from feeder_ae import MNIST_Feeder
import matplotlib.pyplot as plt
from einops import rearrange
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn

# work_dir = './results_dim_20'
work_dir = './workdir/2023-01-11_15-20-21'
dir_en = '{}/encoder.pth.tar'.format(work_dir)
dir_de = '{}/decoder.pth.tar'.format(work_dir)

device = torch.device('cuda:0')
model_en = torch.load(dir_en).to(device)
model_de = torch.load(dir_de).to(device)
test_feeder = MNIST_Feeder(phase='test')
test_dataloader = DataLoader(test_feeder, batch_size=2, shuffle=False, drop_last=False)

def to_image(x_re):
  x_re *= 255
  x_re = x_re.clip(0, 255)
  x_re = x_re.astype(np.int8)
  # x_re = rearrange(x_re, 'C H W -> H W C')
  x_re = Image.fromarray(x_re)
  return x_re

model_en.eval()
model_de.eval()
for p in model_en.parameters():
  p.requires_grad = False
for p in model_de.parameters():
  p.requires_grad = False

eval_iter = tqdm(test_dataloader, dynamic_ncols=True)
# with torch.no_grad():
file_idx = 1
for idx, (xs, xs_dst, ys) in enumerate(eval_iter):
  xs = xs.to(device).float()
  # 加缺损噪声
  xs_noisy = xs.clone()
  xs_noisy[:,:,18:,:] = 0 # 遮挡上半部分
  means, stddevs = model_en(xs.reshape(-1, 28*28))
  means_noisy, stddevs_noisy = model_en(xs_noisy.reshape(-1, 28*28))
  zs = means + torch.randn_like(means)*stddevs
  zs_noisy = means_noisy + torch.randn_like(means)*stddevs_noisy

  # 初始化目标z
  zs_target = zs_noisy.clone() # 以噪声数据的z向量初始化
  # zs_target = torch.randn_like(zs) # 随机初始化
  zs_target.requires_grad = True
  optimizer = Adam([zs_target], lr=0.001)
  loss_re = nn.MSELoss(reduction='sum')
  iteration = 1000
  for i in range(iteration):
    optimizer.zero_grad()
    xs_target_re = model_de(zs_target).reshape(-1, 1, 28, 28)
    # 最小化xs_target和xs_noisy未遮挡的对应部分
    # loss = loss_re(xs_target_re[:,:,:10,:], xs[:,:,:10,:]) / xs.shape[0] + loss_re(xs_target_re[:,:,14:,:], xs[:,:,14:,:]) / xs.shape[0]
    loss = loss_re(xs_target_re[:,:,:18,:], xs[:,:,:18,:]) / xs.shape[0]
    loss.backward()
    if (i+1) % 100 == 0:
      print('iter: {}, loss: {}'.format(i+1, loss.item()))
    optimizer.step()
  
  xs_re = model_de(zs).reshape(-1, 1, 28, 28)
  xs_noisy_re = model_de(zs_noisy).reshape(-1, 1, 28, 28)
  quit = False
  for i in range(xs.shape[0]):
    x_noisy = xs_noisy[i].squeeze().cpu().numpy()
    x = xs[i].squeeze().cpu().numpy()
    y = ys[i]
    x_re = xs_re[i].detach().squeeze().cpu().numpy()
    x_noisy_re = xs_noisy_re[i].detach().squeeze().cpu().numpy()
    x_target_re = xs_target_re[i].detach().squeeze().cpu().numpy()
    image_noisy = to_image(x_noisy)
    image_original = to_image(x)
    image_re = to_image(x_re)
    image_re_noisy = to_image(x_noisy_re)
    image_re_target = to_image(x_target_re)
    
    images = [image_original, image_re, image_noisy, image_re_target]
    titles = ['x', 'x_re', 'x_noisy', 'x_target_re']
    # plt.suptitle('{}'.format(y))
    plt.tight_layout()
    for j in range(len(images)):
      plt.subplot(1, len(images), j+1)
      plt.axis("off")
      plt.title(titles[j])
      plt.imshow(images[j])
    plt.savefig('./{}.jpg'.format(file_idx), bbox_inches='tight', pad_inches=0.1)
    file_idx += 1
    plt.show()

    # cmd = input('q to quit, anything else to continue...')
    # if cmd == 'q':
    if file_idx == 19:
      quit = True
      break
  if quit:
    break