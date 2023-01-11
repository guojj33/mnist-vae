from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time, os, torch, logging, sys
from torchsummary import summary

from vae import Encoder
from vae import Decoder

from feeder_ae import MNIST_Feeder

train_batch_size = 128
eval_batch_size = 128
epoch = 60
lr = 1e-3
resume = False

cur_work_dir = '{}/{}'.format('./workdir', time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(cur_work_dir)
writer = SummaryWriter(cur_work_dir)
def set_logging(cur_work_dir):
  log_format = '[ %(asctime)s ] %(message)s'
  # 在控制台输出
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
  # 在文本输出
  file_handler = logging.FileHandler('{}/output.log'.format(cur_work_dir), mode='w', encoding='UTF-8')
  file_handler.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(file_handler)
set_logging(cur_work_dir)

train_feeder = MNIST_Feeder(phase='train')
test_feeder = MNIST_Feeder(phase='test')
train_dataloader = DataLoader(train_feeder, batch_size=train_batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_feeder, batch_size=eval_batch_size, shuffle=False, drop_last=False)
device = torch.device('cuda:0')

dim_img = 28*28*1
n_hidden = 500
dim_z = 20
keep_prob = 0.99
model_en = Encoder(dim_img, n_hidden, dim_z, keep_prob).to(device)
model_de = Decoder(dim_z, n_hidden, dim_img, keep_prob).to(device)
with open('{}/model.txt'.format(cur_work_dir), 'w') as f:
  f.write(str(model_en))
  f.write(str(model_de))
summary(model_en, (dim_img, ))
summary(model_de, (dim_z, ))

def get_loss(x, x_target):
    batchsz = x.size(0)
    # encoding
    mu, sigma = model_en(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = model_de(z)
    y = torch.clamp(y, 1e-8, 1 - 1e-8) # 0到1

    # loss
    # marginal_likelihood2 = torch.sum(x_target * torch.log(y) + (1 - x_target) * torch.log(1 - y)) / batchsz
    marginal_likelihood = -F.binary_cross_entropy(y, x_target, reduction='sum') / batchsz
    # print(marginal_likelihood2.item(), marginal_likelihood.item())

    KL_divergence = 0.5 * torch.sum(
                                torch.pow(mu, 2) +
                                torch.pow(sigma, 2) -
                                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                               ).sum() / batchsz

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return y, z, loss, marginal_likelihood, KL_divergence

optimizer = Adam(list(model_en.parameters()) + list(model_de.parameters()), lr=lr)
best_eval_loss = None
start_epoch = 0

if resume:
  work_dir = './workdir'
  dirs = os.listdir(work_dir)
  for i, d in enumerate(dirs):
    print('[{}] {}'.format(i, d))

  idx = int(input('idx for dir to load:'))
  dir = '{}/{}/checkpoint.pth.tar'.format(work_dir, dirs[idx])

  checkpoint = torch.load(dir, map_location=device)
  model_en.load_state_dict(checkpoint['encoder'])
  model_de.load_state_dict(checkpoint['decoder'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  start_epoch = checkpoint['epoch']

  logging.info('resume training: loading model from {}'.format(dir))

for e in range(start_epoch, epoch):
  logging.info('Training for Epoch: {}/{}'.format(e+1, epoch))

  model_en.train()
  model_de.train()
  optimizer.zero_grad()

  train_iter = tqdm(train_dataloader, dynamic_ncols=True)
  num_sample = 0
  num_top1 = 0
  train_loss = []
  for idx, (x, x_dst, _) in enumerate(train_iter):
    x = x.float().to(device)
    x_dst = x_dst.float().to(device)

    x = x.reshape(x.shape[0], -1)
    x_dst = x_dst.reshape(x_dst.shape[0], -1)

    y, z, total_loss, loss_likelihood, loss_kl = get_loss(x, x_dst)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    train_loss.append(total_loss.item())

  train_loss = np.array(train_loss)
  train_loss = train_loss.sum() / len(train_loss)
  writer.add_scalar('Loss/Train', train_loss, e)
  writer.add_scalar('Loss_likelihood/Train', loss_likelihood.item(), e)
  writer.add_scalar('Loss_kl/Train', loss_kl.item(), e)
  logging.info('TRAIN loss: {}, loss likelihood: {}, loss kl: {}'.format(train_loss, loss_likelihood.item(), loss_kl.item()))

  logging.info('Evaluating for Epoch: {}/{}'.format(e+1, epoch))

  eval_iter = tqdm(test_dataloader, dynamic_ncols=True)
  with torch.no_grad():
    model_en.eval()
    model_de.eval()
    eval_loss = []
    num_top1 = 0
    num_sample = 0
    for idx, (x, x_dst, _) in enumerate(eval_iter):
      x = x.float().to(device)
      x_dst = x_dst.float().to(device)

      x = x.reshape(x.shape[0], -1)
      x_dst = x_dst.reshape(x_dst.shape[0], -1)

      y, z, total_loss, loss_likelihood, loss_kl = get_loss(x, x_dst)

      eval_loss.append(total_loss.item())

    eval_loss = np.array(eval_loss)
    eval_loss = eval_loss.sum() / len(eval_loss)
    writer.add_scalar('Loss/Eval', eval_loss, e)
    writer.add_scalar('Loss_likelihood/Eval', loss_likelihood.item(), e)
    writer.add_scalar('Loss_kl/Eval', loss_kl.item(), e)
    logging.info('EVAL loss: {}, loss likelihood: {}, loss kl: {}'.format(eval_loss, loss_likelihood.item(), loss_kl.item()))

    def save_model():
      torch.save(model_en, '{}/encoder.pth.tar'.format(cur_work_dir))
      torch.save(model_de, '{}/decoder.pth.tar'.format(cur_work_dir))

    def save_checkpoint():
      checkpoint = {
        'encoder': model_en.state_dict(),
        'decoder': model_de.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': e,
        'best_eval_loss': best_eval_loss,
      }
      name = '{}/checkpoint.pth.tar'.format(cur_work_dir)
      torch.save(checkpoint, name)

    if best_eval_loss:
      if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        save_model()
    else:
      best_eval_loss = eval_loss
      save_model()

    save_checkpoint()

logging.info('Best Eval Loss: {}'.format(best_eval_loss))
writer.close()