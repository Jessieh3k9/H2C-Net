import numpy as np
import os
import natsort
from PIL import Image
from skimage import transform
'''
将 bmp 文件夹中的图像数据处理成 numpy 格式，并保存为 .npy 文件
'''
bmpfile_path='/home/jessica/H2C-Net/OCTA_500/OCTA_6mm/OCT'
npzfile_path='/home/jessica/H2C-Net/OCTA_500/OCTA_6mm/OCT_npy'
ctlist = os.listdir(bmpfile_path)
ctlist = natsort.natsorted(ctlist)
for ct in ctlist:
     data = []
     bscanlist=os.listdir(os.path.join(bmpfile_path,ct))
     bscanlist = natsort.natsorted(bscanlist)
     for bscan in bscanlist:
         data.append(np.array(Image.open(os.path.join(bmpfile_path,ct,bscan)).resize((400,128))))
     np.save(os.path.join(npzfile_path,ct),data)

bmpfile_path='/home/jessica/H2C-Net/OCTA_500/OCTA_6mm/OCTA'
npzfile_path='/home/jessica/H2C-Net/OCTA_500/OCTA_6mm/OCTA_npy'
ctlist = os.listdir(bmpfile_path)
ctlist = natsort.natsorted(ctlist)
for ct in ctlist:
     data = []
     bscanlist=os.listdir(os.path.join(bmpfile_path,ct))
     bscanlist = natsort.natsorted(bscanlist)
     for bscan in bscanlist:
         data.append(np.array(Image.open(os.path.join(bmpfile_path, ct, bscan)).resize((400, 128))))
     np.save(os.path.join(npzfile_path,ct),data)

#bmpfile_path='/home/jessica/H2C-Net/OCTA_500/OCTA_3mm/OCT'
#npzfile_path='/home/jessica/H2C-Net/OCTA_500/OCTA_3mm/OCT_npy'
#ctlist = os.listdir(bmpfile_path)
#ctlist = natsort.natsorted(ctlist)
#for ct in ctlist:
#    data = []
#    bscanlist=os.listdir(os.path.join(bmpfile_path,ct))
#    bscanlist = natsort.natsorted(bscanlist)
#    for bscan in bscanlist:
#        data.append(np.array(Image.open(os.path.join(bmpfile_path, ct, bscan)).resize((304, 128))))
#    np.save(os.path.join(npzfile_path,ct),data)

#bmpfile_path='/home/jessica/H2C-Net/OCTA_500/OCTA_3mm/OCTA'
#npzfile_path='/home/jessica/H2C-Net/OCTA_500/OCTA_3mm/OCTA_npy'
#ctlist = os.listdir(bmpfile_path)
#ctlist = natsort.natsorted(ctlist)
#for ct in ctlist:
#    data = []
#    bscanlist=os.listdir(os.path.join(bmpfile_path,ct))
#    bscanlist = natsort.natsorted(bscanlist)
#    for bscan in bscanlist:
#        data.append(np.array(Image.open(os.path.join(bmpfile_path, ct, bscan)).resize((304, 128))))
#    np.save(os.path.join(npzfile_path,ct),data)