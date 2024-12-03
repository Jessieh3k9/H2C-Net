import os
import numpy as np
import sys  # 导入sys模块
sys.setrecursionlimit(30000)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numba as nb
from numba import jit


def dice_coefficient_per_class(pred, target, class_labels=[0,1,2,3]):
    classnum = target.max()
    print(classnum)
    dice_scores = np.zeros((int(classnum), 1))
    for class_label in range(int(classnum)):
        pred_class = (torch.from_numpy(pred) == (class_label+1))
        target_class = (target == (class_label+1))

        intersection = torch.sum(pred_class * target_class)
        union = torch.sum(pred_class) + torch.sum(torch.from_numpy(target_class))

        dice = (2.0 * intersection) / (union + 1e-5)  # 添加平滑项
        dice_scores[class_label] = dice.item()

    return dice_scores

        

@nb.njit
def dfs(i, j, label, matrix):
    n, m = matrix.shape
    matrix_writable = matrix.copy()  # 创建一个可写的副本
    stack = [(i, j)]
    while stack:
        i, j = stack.pop()
        if i < 0 or i >= n or j < 0 or j >= m or matrix_writable[i, j] != label:
            continue
        matrix_writable[i, j] = -1
        stack.append((i - 1, j))
        stack.append((i + 1, j))
        stack.append((i, j - 1))
        stack.append((i, j + 1))
    return matrix_writable

def connectionEfficient(matrix):
    component_counts = {}
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            label = matrix[i, j]
            if label in (2, 3) and label not in component_counts:
                component_counts[label] = 0
            if label in (2, 3):
                component_counts[label] += 1
                matrix = dfs(i, j, label, matrix)
    return component_counts

class connectLoss(nn.Module):
    
    def __init__(self):
        super(connectLoss, self).__init__()

    def forward(self, pred, anno):
        pred_argmax = torch.argmax(pred, dim=1)
        pred_argmax = pred_argmax.detach()
        annoEF = 0
        predEF = 0
        connEF = 0
        for i in range(anno.shape[0]):
            annoEF = connectionEfficient(anno[i,0,:,:].cpu().numpy())
            predEF = connectionEfficient(pred_argmax[i,0,:,:].cpu().numpy())
            connEF += ((1-(2*min(predEF[2],annoEF[2])/(annoEF[2]+predEF[2])))+(1-(2*min(predEF[3],annoEF[3])/(annoEF[3]+predEF[3]))))
        loss = torch.tensor(connEF/2,requires_grad=True)
        # print(loss)
        return loss
def check_dir_exist(dir):
    """create directories"""
    if os.path.exists(dir):
        return
    else:
        names = os.path.split(dir)
        dir = ''
        for name in names:
            dir = os.path.join(dir,name)
            if not os.path.exists(dir):
                try:
                    os.mkdir(dir)
                except:
                    pass
        print('dir','\''+dir+'\'','is created.')

def cal_Dice(img1,img2):
    shape = img1.shape
    I = 0
    U = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] >= 1 and img2[i,j] >= 1:
                I += 1
            if img1[i,j] >= 1 or img2[i,j] >= 1:
                U += 1
    return 2*I/(I+U+1e-5)


def cal_acc(img1,img2):
    shape = img1.shape
    acc = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] == img2[i,j]:
                acc += 1
    return acc/(shape[0]*shape[1])
tup1 = ('背景','毛细血管','红色','蓝色','FAZ') 
def cal_miou(img1,img2):
    classnum = img2.max()
    iou=np.zeros((int(classnum),1))
    for i in range(int(classnum)):
        imga=img1==i+1
        imgb=img2==i+1
        imgi=imga * imgb
        imgu=imga + imgb
        iou[i]=np.sum(imgi)/np.sum(imgu)
        # print(tup1[i],"_IoU:{}".format(iou[i]))
    # print("iou",iou)
    miou=np.mean(iou)
    # print("miou",miou)
    return miou

def cal_cavfIOU(img1,img2):
    classnum = img2.max()
    iou=np.zeros((int(classnum),1))
    for i in range(int(classnum)):
        imga=img1==i+1
        imgb=img2==i+1
        imgi=imga * imgb
        imgu=imga + imgb
        iou[i]=np.sum(imgi)/np.sum(imgu)
    return iou

def make_one_hot(input, shape):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    result = torch.zeros(shape)
    result.scatter_(1, input.cpu(), 1)
    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1).cpu()
        target = target.contiguous().view(target.shape[0], -1).cpu()

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        shape=predict.shape
        # print(shape)
        target = torch.unsqueeze(target, 1)
        target=make_one_hot(target.long(),shape)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]

def get_patch_random(data,label, cube_size, patch_size):
    patch_pos = []
    for i in range(3):
        patch_pos.append(torch.randint(0, cube_size[i] - patch_size[i] + 1, (1,)))
        #print(cube_size[i], patch_size[i], patch_pos[i])
    data_crop = data[:, :, patch_pos[0]:patch_pos[0] + patch_size[0], patch_pos[1]:patch_pos[1] + patch_size[1],patch_pos[2]:patch_pos[2] + patch_size[2]]
    data_crop = data_crop.contiguous()
    label_crop = label[:, :, patch_pos[1]:patch_pos[1] + patch_size[1],patch_pos[2]:patch_pos[2] + patch_size[2]]
    label_crop  = label_crop.contiguous()
    return data_crop,label_crop

def split_test(data,model, cube_size, patch_size,n_classes):
    outshape=[1,n_classes,1,cube_size[1],cube_size[2]]
    result = torch.zeros(outshape)
    result = result.to(data.device)
    for x in range(0, cube_size[0], patch_size[0]):
        for y in range(0, cube_size[1], patch_size[1]):
            for z in range(0, cube_size[2], patch_size[2]):
                input=data[:,:,x:x+patch_size[0],y:y+patch_size[1],z:z+patch_size[2]]
                output=model(input)
                result[:,:,x:x+patch_size[0],y:y+patch_size[1],z:z+patch_size[2]]=output
    return result

def dice_coefficient_per_class(pred, target, class_labels=[0,1,2,3]):
    classnum = target.max()
    dice_scores = np.zeros((int(classnum), 1))
    for class_label in range(int(classnum)):
        pred_class = (torch.from_numpy(pred) == (class_label+1))
        target_class = (target == (class_label+1))

        intersection = torch.sum(pred_class * target_class)
        union = torch.sum(pred_class) + torch.sum(torch.from_numpy(target_class))

        dice = (2.0 * intersection) / (union + 1e-5)  # 添加平滑项
        dice_scores[class_label] = dice.item()

    return dice_scores
if __name__ == '__main__':
    pred = torch.tensor([[0, 1, 1, 2], [2, 1, 0, 2]])

    pred_class=(pred==1)
    print(pred_class)