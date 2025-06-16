import os
import copy
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torchvision import utils as vutils
from tqdm import tqdm
from DatasetLoader import Loader
from utils import AverageMeter, calc_psnr
import matplotlib.pyplot as plt
import numpy as np
from unet import UNet


if __name__ == '__main__':
    outputs_dir_base = './output/'
    cnt = 0
    data_path = 'F:/dataset/band1To3-50%/train/target/'
    val_path = data_path.replace('train', 'val')

    start_epoch = 0
    num_epochs = 200
    lr = 1e-3
    batch_size = 2
    epochTrainLoss = []
    epochValLoss = []
    epochPSNR = []

    seed = 100
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    outputs_dir = os.path.join(outputs_dir_base, '{}'.format('S2F'))
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    cudnn.benchmark = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = UNet().to(device)
    cri1 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    best_loss = 1

    train_dataset = Loader(data_path)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True)

    eval_dataset = Loader(val_path)
    eval_dataloader = DataLoader(dataset=eval_dataset, shuffle=False, batch_size=1)

    # checkpoint = torch.load(os.path.join(outputs_dir, 'epoch_s.pkl'))
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch']
    # start_epoch = start_epoch + 1

    for epoch in range(start_epoch, num_epochs):
        epochnow = epoch
        model.train()
        epoch_loss = AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data.to(device)

                preds = model(inputs)

                loss = cri1(preds, labels)
                epoch_loss.update(loss.item(), len(inputs))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_loss.avg))
                t.update(len(inputs))
            epochTrainLoss.append(epoch_loss.avg)

        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
                os.path.join(outputs_dir, 'epoch_d.pkl'))
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
                os.path.join(outputs_dir, 'epoch_s.pkl'))

        for params in optimizer.param_groups:
            print(params['lr'])

        if epoch % 20 == 0 and epoch != 0:
            for params in optimizer.param_groups:
                # Decays the learning rate of each parameter group by gamma(0.5) every step_size(20) epochs.
                params['lr'] *= 0.5

        model.eval()
        epoch_psnr = AverageMeter()
        epoch_lossval = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)

            lossval = cri1(preds, labels)
            epoch_lossval.update(lossval.item(), len(inputs))
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        epochValLoss.append(epoch_lossval.avg)
        epochPSNR.append(epoch_psnr.avg)
        print('eval psnr: {:.6f}'.format(epoch_psnr.avg))
        print('eval loss: {:.6f}'.format(epoch_lossval.avg))

        if epoch_lossval.avg <= best_loss:
            best_epoch = epoch
            best_loss = epoch_lossval.avg
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(outputs_dir, 'best.pth'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
                os.path.join(outputs_dir, 'best.pkl'))
            print('Save the model!')

        # If the validation loss doesn't improve within 20 epochs, stop training.
        if epoch >= 30:
            if epoch_lossval.avg > best_loss:
                cnt = cnt + 1
            else:
                cnt = 0
        if cnt == 20:
            cnt = 0
            break

    print('best epoch: {}, psnr: {:.4f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(outputs_dir, 'best.pth'))

    # Draw the figure of Traning Loss and Validation Loss
    plt.figure()
    x = list(range(start_epoch, epochnow+1))
    plt.plot(x, epochTrainLoss)
    plt.plot(x, epochValLoss)
    plt.legend(['trainLoss', 'valLoss'])
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(outputs_dir, 'loss.png'))
    plt.show()

    # Draw the figure of PSNR in Validation
    plt.figure()
    plt.plot(x, epochPSNR)
    plt.xlabel('epochs')
    plt.ylabel('PSNR')
    plt.savefig(os.path.join(outputs_dir,'psnr.png'))
    plt.show()


