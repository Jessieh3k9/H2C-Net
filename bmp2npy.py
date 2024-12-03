import numpy as np
import os
import natsort
from PIL import Image
from skimage import transform

# bmpfile_path='/raid5/octa_500/OCTA_6mm/OCT'
# npzfile_path='/raid5/octa_500/OCTA_6mm/OCT_npy'
# ctlist = os.listdir(bmpfile_path)
# ctlist = natsort.natsorted(ctlist)
# for ct in ctlist:
#     data = []
#     bscanlist=os.listdir(os.path.join(bmpfile_path,ct))
#     bscanlist = natsort.natsorted(bscanlist)
#     for bscan in bscanlist:
#         data.append(np.array(Image.open(os.path.join(bmpfile_path,ct,bscan)).resize((400,128))))
#     np.save(os.path.join(npzfile_path,ct),data)

# bmpfile_path='/raid5/octa_500/OCTA_6mm/OCTA'
# npzfile_path='/raid5/octa_500/OCTA_6mm/OCTA_npy'
# ctlist = os.listdir(bmpfile_path)
# ctlist = natsort.natsorted(ctlist)
# for ct in ctlist:
#     data = []
#     bscanlist=os.listdir(os.path.join(bmpfile_path,ct))
#     bscanlist = natsort.natsorted(bscanlist)
#     for bscan in bscanlist:
#         data.append(np.array(Image.open(os.path.join(bmpfile_path, ct, bscan)).resize((400, 128))))
#     np.save(os.path.join(npzfile_path,ct),data)

bmpfile_path='/raid5/octa_500/OCTA_3mm/OCT'
npzfile_path='/raid5/octa_500/OCTA_3mm/OCT_npy'
ctlist = os.listdir(bmpfile_path)
ctlist = natsort.natsorted(ctlist)
for ct in ctlist:
    data = []
    bscanlist=os.listdir(os.path.join(bmpfile_path,ct))
    bscanlist = natsort.natsorted(bscanlist)
    for bscan in bscanlist:
        data.append(np.array(Image.open(os.path.join(bmpfile_path, ct, bscan)).resize((304, 128))))
    np.save(os.path.join(npzfile_path,ct),data)

bmpfile_path='/raid5/octa_500/OCTA_3mm/OCTA'
npzfile_path='/raid5/octa_500/OCTA_3mm/OCTA_npy'
ctlist = os.listdir(bmpfile_path)
ctlist = natsort.natsorted(ctlist)
for ct in ctlist:
    data = []
    bscanlist=os.listdir(os.path.join(bmpfile_path,ct))
    bscanlist = natsort.natsorted(bscanlist)
    for bscan in bscanlist:
        data.append(np.array(Image.open(os.path.join(bmpfile_path, ct, bscan)).resize((304, 128))))
    np.save(os.path.join(npzfile_path,ct),data)