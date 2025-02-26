from absl import app
import wandb
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config import FLAGS
from src.utils.model import preload_models
from src.utils.dataset import TaRF_RGB_NOCS_DataLoader
from src.utils.train import train
from src.utils.validation import validation


class psnr_loss(nn.Module):
    def __init__(self):
        super(psnr_loss, self).__init__()
        
    def forward(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return -(20 * torch.log10(1.0 / torch.sqrt(mse)))


def get_current_datetime():
    # Get the current date and time
    current_datetime = datetime.now()
    # Format it as needed, e.g., "YYYY-MM-DD HH:MM:SS"
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
    return formatted_datetime


def main(argv):
    # 0. Initialization
    torch.manual_seed(FLAGS.seed)
    starting_time = get_current_datetime()
    
    if FLAGS.wandb:
        wandb.init(project=FLAGS.wandb_project, group=FLAGS.wandb_group, name=starting_time, config=FLAGS)
        wandb.config.update(FLAGS)
    
    # 1. Load data
    # data = Loader(FLAGS)
    # train_loader = DataLoader(data, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    
    if FLAGS.use_loader == 'nocs':
        train_data = TaRF_RGB_NOCS_DataLoader(FLAGS, train=True)
    else:
        raise ValueError("Invalid loader")
    train_loader = DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True, num_workers=24)
    
    # 2. Define models
    models = preload_models(flags=FLAGS,conditioning_shape=train_data.get_conditioning_shape())
    
    # 3. Define optimizer
    optimizer = optim.Adam(models['diffusion'].parameters(), lr=FLAGS.lr, weight_decay=1e-10)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120, 160], gamma=0.1)
    
    # 4. Define loss function
    loss_1 = nn.MSELoss()
    loss_2 = nn.L1Loss()
    psnr_loss_fn = psnr_loss()
    loss_fn = loss_1
    
    # 5. Train model
    if not(FLAGS.test):
        train(models, optimizer, scheduler, train_loader, loss_fn, starting_time, FLAGS)
    else:
        test_data = TaRF_RGB_NOCS_DataLoader(FLAGS, train=False)
        test_loader = DataLoader(test_data, batch_size=FLAGS.batch_size, shuffle=False, num_workers=24)
        validation(models, train_loader, starting_time, FLAGS)
    
    
    
if __name__ == '__main__':
    app.run(main)