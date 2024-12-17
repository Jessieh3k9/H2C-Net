from options.test_options import TestOptions
import numpy as np
import cv2
import torch
import os
from tqdm import tqdm
import torch.nn as nn
import BatchDataReader  #使用 BatchDataReader.CubeDataset 加载测试数据集
from prefetch_generator import BackgroundGenerator

#通过 directMAXpool 层对数据进行降采样 高度缩小1/32
class directMAXpool(nn.Module):
    #ks表示几个压在一起 
    def __init__(self):
        super().__init__()
        # self.maxpool=nn.MaxPool3d(kernel_size=[23,1,1],stride=[7,1,1])
        self.maxpool=nn.MaxPool3d(kernel_size=[32,1,1],stride=[32,1,1])
    def forward(self,x):
        x = self.maxpool(x)
        return x
"""
make_one_hot()将类别索引转换为独热编码
将类别索引转换为独热编码的目的主要是与模型的训练机制兼容，支持多类别和多标签任务，
提高计算灵活性，并避免类别之间的错误关系假设。这种表示方法对分类、分割和其他任务的实现至关重要。        
"""
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
"""
拆分拼接(UA+C)
"""
def split_test(data,batch_size, data_size): #1，1，128*304*304 
    outshape=[batch_size,1,data_size*2,data_size*2] #(batch_size, 1, 256, 608)

    result = torch.zeros(outshape)
    result = result.to(data.device)
    
    i=0
    for x in range(0, data_size*2, data_size):
        for y in range(0, data_size*2, data_size):
            input = data[:,0,i,0:data_size,0:data_size]
            result[:,0,x:x+data_size,y:y+data_size]=input
            i=i+1
    return result
"""
代码用途
将标注张量（类别索引）转换为可视化图像，用颜色直观表达。
常用于分割任务中，验证标注的正确性或可视化模型预测结果。
    
"""
def create_visual_anno(anno):
    """将标注张量 anno 转换为一个可视化图像，使用不同的颜色表示不同的类别"""
    # assert np.max(anno) <= 7, "only 7 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [53, 32, 15],
        1: [143, 165, 171],  # cornsilk
        2: [28, 25, 215],  # cornflowerblue
        3: [186, 131, 43],  # mediumAquamarine
        4: [106, 217, 166],  # peru
    }
    # visualize
    print("anno.shape",anno.shape)
    shape =anno.shape[2]
    visual =  np.zeros((400, 400, 3))
    visual_anno = np.zeros((1600, 1600, 3))
    for i in range(shape):  # i for h # 遍历高度
        for j in range(shape):        # 遍历宽度
            # print(label2color_dict[anno[0,i,j]])
            color = label2color_dict[anno[0,i, j]]  # 查找当前像素的颜色
            visual[i, j, 0] = color[0]   # 红色通道
            visual[i, j, 1] = color[1]   # 蓝色通道
            visual[i, j, 2] = color[2]   # 绿色通道
    print(visual.shape)
    for x in range(0, shape*4, shape):      #遍历大图的高度
        for y in range(0, shape*4, shape):  #遍历大图的宽度
            visual_anno[x:x+shape,y:y+shape,:]=visual
            i=i+1
    return visual_anno

def create_visual_predict(predict):
    """"""
    # assert np.max(anno) <= 7, "only 7 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [53, 32, 15],
        1: [143, 165, 171],  # cornsilk
        2: [28, 25, 215],  # cornflowerblue
        3: [186, 131, 43],  # mediumAquamarine
        4: [106, 217, 166],  # peru
    }
    # visualize
    print("anno.shape",predict.shape)
    shape =predict.shape[2]
    visual =  np.zeros((1600, 1600, 3))
    # visual_anno = np.zeros((1600, 1600, 3))
    for i in range(shape):  # i for h # 遍历高度
        for j in range(shape):# 遍历宽度
            print(predict)
            # print(label2color_dict[anno[0,i,j]])
            color = label2color_dict[predict[0,i, j]]  # 查找当前像素的颜色
            visual[i, j, 0] = color[0]   # 红色通道
            visual[i, j, 1] = color[1]   # 绿色通道
            visual[i, j, 2] = color[2]   # 蓝色通道
    return visual
# def split_test(data,batch_size, data_size):
#     outshape=[batch_size,1,data_size*4,data_size*4]
#     result = torch.zeros(outshape)
#     result = result.to(data.device)
#     i=0
#     for x in range(0, data_size*4, data_size):
#         for y in range(0, data_size*4, data_size):
#             input = data[:,0,i,0:data_size,0:data_size]
#             result[:,0,x:x+data_size,y:y+data_size]=input
#             i=i+1
#     return result

opt = TestOptions().parse()
# BGR=np.zeros((opt.data_size[0],opt.data_size[1],3))
test_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.test_ids, opt.data_size, opt.modality_filename,opt.label_filename,is_dataaug=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
# test1 = cv2.imread('/raid5/octa_500/OCTA_6mm/Projection Maps/GT_MultiTask/10021.bmp', cv2.IMREAD_GRAYSCALE).astype(np.float32)
# print(test_dataset)
pbar = tqdm(enumerate(BackgroundGenerator(test_loader)), total=len(test_loader))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dirct =directMAXpool()
BGR=np.zeros((1600,1600,3))
for itr, (test_images, test_annotations,cubename) in pbar:
    print(test_images.shape)
    test_images = dirct(test_images)
    
    test_images=split_test(test_images,1,test_images.shape[3]).numpy()
    print(type(test_images))
    cv2.imwrite(os.path.join(opt.saveroot, 'test', cubename[0]), test_images[0,0, :, :])
    # print("test_images111.shape",test_images.shape)
    # test_images=torch.squeeze(test_images,0)
    # BGR=create_visual_predict(test_images)
    # cv2.imwrite(os.path.join(opt.saveroot, 'test', cubename[0]), BGR)
    
    # print(test_annotations.size(),cubename)
    # test_annotations=torch.squeeze(test_annotations,0)
    # train_annotations = test_annotations.to(device=device, dtype=torch.long)
    # print("train_annotations.shape",train_annotations.shape)
    # pred_softmax = train_annotations.cpu().detach().numpy() 
    # print("pred_softmax.shape",pred_softmax.shape)
    # BGR = create_visual_anno(pred_softmax)
    
    
    # print(BGR)
    # print("BGR.shape",BGR.shape)
    # train_annotations = train_annotations.unsqueeze(1)
    # print("train_annotations",train_annotations.shape)
    # print(train_annotations.size(),cubename)
    # pred_softmax = torch.nn.functional.softmax(train_annotations, dim=1)
    # pred_softmax = pred_softmax.cpu().detach().numpy()
    # target = torch.unsqueeze(train_annotations, 1)
    # shape =([16,5,400,400])
    # target=make_one_hot(target.long(),shape)
    # t = train_annotations.repeat(4,1,1)  
    # print(target.size())                                                   
    # BGR[:,:,0]=53*train_annotations[0,0,:,:]+143*train_annotations[0,1,:,:]+28*train_annotations[0,2,:,:]+186*train_annotations[0,3,:,:]+106*train_annotations[0,4,:,:]
    # BGR[:,:,1]=32*train_annotations[0,0,:,:]+165*train_annotations[0,1,:,:]+25*train_annotations[0,2,:,:]+131*train_annotations[0,3,:,:]+217*train_annotations[0,4,:,:]
    # BGR[:,:,2]=15*train_annotations[0,0,:,:]+171*train_annotations[0,1,:,:]+215*train_annotations[0,2,:,:]+43*train_annotations[0,3,:,:]+166*train_annotations[0,4,:,:]
    # cv2.imwrite(os.path.join(opt.saveroot, 'test', cubename[0]), BGR)
# test = test1[test1>0]
# print(np.unique(test))
# print(test)
# if (test1==0).all():
#     print("quanbudouyu0")
# else:
#     print("1")


