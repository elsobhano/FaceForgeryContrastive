import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
import os
import datatime
import time
import gc

from hyperparams import hypers
from components.Data.dataingestion import DataIngest
from components.Data.dataloader import DataIngestConfig, MyDataset
from components.model_core.model import Two_Stream_Net
from components.Losses.ConLoss import SupConLoss
from src.utils import AverageMeter, save_model


def save_path(hypers):

    model_path = './save/SupCon/{}_models'.format(hypers.get('dataset'))
    tb_path = './save/SupCon/{}_tensorboard'.format(hypers.get('dataset'))
    lr_decay_epochs = [int(it) for it in hypers.get('lr_decay_epochs').split(',')]


    save_time = str(datetime.datetime.now())
    model_name = '{}_{}_{}_{}_lr_{}_decay_{}_epochs_{}_bsz_{}_temp_{}'.\
            format(save_time, hypers.get('method'), hypers.get('dataset'), hypers.get('model'), hypers.get('learning_rate'),
                hypers.get('weight_decay'), hypers.get('epochs') , hypers.get('batch_size'), hypers.get('temp'))

    tb_folder = os.path.join(tb_path, model_name)
    if not os.path.isdir(tb_folder):
        os.makedirs(tb_folder)

    save_folder = os.path.join(model_path, model_name)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    return save_folder, tb_folder

def set_loader():

    Config = DataIngestConfig()
    DF_train_dataset = MyDataset(Config.DF_train_data)
    F2F_train_dataset = MyDataset(Config.F2F_train_data)
    FS_train_dataset = MyDataset(Config.FS_train_data)
    OR_train_dataset = MyDataset(Config.OR_train_data)

    train_dataset = torch.utils.data.ConcatDataset([DF_train_dataset, F2F_train_dataset, FS_train_dataset, OR_train_dataset, OR_train_dataset, OR_train_dataset])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hypers.get('batch_size'), shuffle=True)
    return train_loader

def set_model(hypers, device):
    
    gc.collect()
    torch.cuda.empty_cache()
    model = Two_Stream_Net()
    criterion = SupConLoss(temperature=hypers.get('temp'))
    model = model.to(device)
    criterion = criterion.cuda()

    return model, criterion

def set_optimizer(model, hypers):
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=learning_rate,
    #                       momentum=momentum,
    #                       weight_decay=weight_decay)
    optimizer = optim.Adamax(model.parameters(),  lr=1e-3, eps=1e-4, weight_decay=1e-4)

    return optimizer

def train(train_loader, model, criterion, optimizer, epoch, hypers, device):
    """one epoch training"""
    

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        images = images.to(device)
        labels = labels.to(device)
        bsz = labels.shape[0]

        # warm-up learning rate
#         warmup_learning_rate(epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
#         print(features.shape, labels.shape)
#         break
        if hypers.get('method') == 'SupCon':
            loss = criterion(features, labels)
        elif hypers.get('method') == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(hypers.get('method')))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % hypers.get('print_freq') == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main(hypers, device):
    
    # build model and criterion
    model, criterion = set_model(hypers, device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of Network's Parameters: {total_params:,}")

    # build optimizer
    optimizer = set_optimizer(model, hypers)
    train_loader = set_loader()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                               max_lr=1e-3, 
                               epochs=hypers.get('epochs'),
                               steps_per_epoch=len(train_loader),
                               pct_start=16.0/hypers.get('epochs'),
                               div_factor=25,
                               final_div_factor=2)
    model.train()

    save_folder, tb_folder = save_path(hypers)
    # tensorboard
    writer = SummaryWriter(tb_folder)
    print('Start Training')
    # training routine
    for epoch in range(1, hypers.get('epochs')):
#         adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
#         return 0
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        print('epoch {}, loss {:.2f}'.format(epoch, loss))
        print('*'*15)
        
        # tensorboard logger
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % hypers.get('save_freq') == 0:
            save_file = os.path.join(
                save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, epoch, save_file)
    
    writer.flush()
    writer.close()

    # save the last model
    save_file = os.path.join(
        save_folder, 'last.pth')
    save_model(model, optimizer, hypers.get('epochs'), save_file)


if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(hypers, device=device)