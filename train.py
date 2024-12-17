import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
import model as model
import time
#from init_weights import init_weight
import utils
import shutil
import natsort
from options.train_options import TrainOptions
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import BatchDataReader
import scipy.io as io
from torchsummary import summary

torch.backends.cudnn.benchmark = False
def train_net(net,device):
    #train setting
    val_num = opt.val_ids[1] - opt.val_ids[0]
    DATA_SIZE = opt.data_size
    best_valid_miou=0
    # save_loss=[]
    model_save_path = os.path.join(opt.saveroot, '/home/jessica/H2C-Net/logs/check_points6m/check_points_pmodel2')
    best_model_save_path = os.path.join(opt.saveroot, '/home/jessica/H2C-Net/logs/check_points6m/best_model_pmodel2')
    # Read Data
    print("Start Setup dataset reader")
    train_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.train_ids,opt.data_size,opt.modality_filename,opt.label_filename,is_dataaug=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    valid_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.val_ids, opt.data_size, opt.modality_filename,opt.label_filename,is_dataaug=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    # Setting Optimizer
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)
    # optimizer = Lion(net.parameters(), opt.lr)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max =opt.num_epochs*140,eta_min=0.0001,)
    #Setting Loss 

    Loss_CE = nn.CrossEntropyLoss()
    Loss_DSC= utils.DiceLoss()
    
    #Start train
    for epoch in tqdm(range(1, opt.num_epochs + 1)):
        net.train()
        valid_mloss=0
        train_losssum=0
        pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader))
        for itr, (train_images, train_annotations,cubename) in pbar:
            train_images =train_images.to(device=device, dtype=torch.float32)
            train_annotations = train_annotations.to(device=device, dtype=torch.long)
            pred= net(train_images)
            loss = Loss_CE(pred, train_annotations)+ Loss_DSC(pred, train_annotations)
            train_losssum=loss+train_losssum
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # save_loss.append(loss.item())
            
            #print(loss.item())
        print("lr: %f",(epoch, optimizer.param_groups[0]['lr']))
        #Start Val
        with torch.no_grad():
            
            #Calculate validation mIOU
            val_miou_sum = 0
            val_acc_sum = 0                                                                     
            val_dice_sum = 0
            net.eval()
            pbar = tqdm(enumerate(BackgroundGenerator(valid_loader)), total=len(valid_loader))
            for itr, (test_images, test_annotations,cubename) in pbar:
                result = np.zeros((DATA_SIZE[1], DATA_SIZE[2]))
                test_images = test_images.to(device=device, dtype=torch.float32)
                test_annotations_1 = np.squeeze(test_annotations).cpu().detach().numpy()
                test_an = test_annotations.to(device=device, dtype=torch.long)
                pred=net(test_images)
                pred_argmax = torch.argmax(pred, dim=1)
                valid_loss = Loss_CE(pred,test_an)+Loss_DSC(pred, test_an)
                valid_mloss = valid_mloss +valid_loss
                result= np.squeeze(pred_argmax).cpu().detach().numpy()
                val_miou_sum+= utils.cal_miou(result,test_annotations_1)
                val_acc_sum += utils.cal_acc(result, test_annotations_1)
                val_dice_sum += utils.cal_Dice(result, test_annotations_1)
            val_miou=val_miou_sum/val_num
            val_dice = val_dice_sum /val_num
            val_acc = val_acc_sum / val_num
            print("Step:{}, Valid_mIoU:{}".format(epoch, val_miou))
            print("Step:{}, val_dice:{}".format(epoch, val_dice))
            print("Step:{}, val_acc:{}".format(epoch, val_acc))
            #Save model
            checkpointName = '{:.6f}'.format(val_miou) + '-' + f'{epoch}.pth'
            torch.save(net.state_dict(), os.path.join(model_save_path, checkpointName))
            logging.info(f'Checkpoint {epoch} saved!')

            #checkpointName = '{:.6f}'.format(val_miou) + '-' +f'{epoch}.pth'
            #torch.save(net.module.state_dict(),os.path.join(model_save_path,checkpointName))
            #logging.info(f'Checkpoint {epoch} saved !')
            #save best model
            if val_miou > best_valid_miou:
                temp = '{:.6f}'.format(val_miou)
                os.mkdir(os.path.join(best_model_save_path,temp))
                temp2= f'{epoch}.pth'
                shutil.copy(os.path.join(model_save_path,checkpointName),os.path.join(best_model_save_path,temp,temp2))
                model_names = natsort.natsorted(os.listdir(best_model_save_path))
                if len(model_names) == 4:
                    shutil.rmtree(os.path.join(best_model_save_path,model_names[0]))
                best_valid_miou = val_miou
    # io.savemat(os.path.join(opt.saveroot, 'loss.mat'),{'loss':save_loss})

        

if __name__ == '__main__':
    #tensorboard --inspect --logdir
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    #loading options
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    opt = TrainOptions().parse()
    # #setting GPU
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '5692'
    # torch.distributed.init_process_group('nccl',world_size=1,rank=0)

    
    #选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = model.H2C_Net(in_channels=opt.in_channels, n_classes=opt.n_classes)
    #net.apply(model.init_weight)
    # torch.cuda.empty_cache()
    
    
    #load trained model
    if opt.load:
        net.load_state_dict(
            torch.load(opt.load, map_location=device)
        )
        logging.info(f'Model loaded from {opt.load}')
    #input the model into GPU
    net.to(device=device)
    # net=net.cuda()
    # net=torch.nn.DataParallel(net,[0,1]).cuda()
    # device = torch.device("cuda:1,2")
    # net.to(device)
    try:
        train_net(net=net,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
            





