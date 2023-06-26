# -*- coding: utf-8 -*-
# Equivalent to the test.py folder in the baseline code
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, PCB, PCB_test
from utils import fuse_all_conv_bn
#fp16
if __name__ == '__main__':    
    try:
        from apex.fp16_utils import *
    except ImportError: # will be 3.x series
        print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
    ######################################################################
    # Options
    # --------
    # data_dir = 'C:\\Users\\Mavara\\Desktop'
    # query='D1_test\\frame_sequence_normal\\D1'
    # gallery='D3_test\\frame_sequence_normal\\D3'
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--which_epoch',default='59', type=str, help='0,1,2,3...or last')
    parser.add_argument('--data_dir',default='C:\\Users\\Mavara\\Desktop',type=str, help='main data directory')
    parser.add_argument('--query',default='D1_test\\frame_sequence_normal\\D1',type=str, help='path of query folder')
    parser.add_argument('--gallery',default='D3_test\\frame_sequence_normal\\D3',type=str, help='path of gallery folder')
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
    parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
    parser.add_argument('--ibn', action='store_true', help='use ibn.' )
    parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'
    print('device:',device)
    
    opt = parser.parse_args()
    ###load config###
    # load the training config
    data_dir = opt.data_dir
    query = opt.query
    gallery = opt.gallery
    config_path = os.path.join('./model',opt.name,'opts.yaml')
    with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'

    opt.stride = config['stride']
    if 'nclasses' in config: # tp compatible with old config files
        opt.nclasses = config['nclasses']
    else: 
        opt.nclasses = 15 

    if 'ibn' in config:
        opt.ibn = config['ibn']
    if 'linear_num' in config:
        opt.linear_num = config['linear_num']

    str_ids = opt.gpu_ids.split(',')
    which_epoch = opt.which_epoch
    name = opt.name
    test_dir = opt.test_dir

    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >=0:
            gpu_ids.append(id)

    print('We use the scale: %s'%opt.ms)
    str_ms = opt.ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    # set gpu ids
    if len(gpu_ids)>0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    ######################################################################
    # Load Data
    # ---------
    #
    # We will use torchvision and torch.utils.data packages for loading the
    # data.
    #
    h, w = 256, 128

    data_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
    ])


    # data_dir = 'C:\\Users\\Mavara\\Desktop'
    # query='D1_test\\frame_sequence_normal\\D1'
    # gallery='D3_test\\frame_sequence_normal\\D3'

    data_dir = 'C:\\Users\\ASUS\\Desktop\\fruit\\crop'
    query='fruit10'
    gallery='fruit12'

    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in [gallery,query]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                shuffle=False, num_workers=1) for x in [gallery,query]}                                     
    class_names = image_datasets[query].classes
    use_gpu = torch.cuda.is_available()
    use_gpu = False
    ######################################################################
    # Load model
    #---------------------------
    def load_network(network):
        save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
        network.load_state_dict(torch.load(save_path))
        return network


    ######################################################################
    # Extract feature
    # ----------------------
    #
    # Extract feature from  a trained model.
    #
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def extract_feature(model,dataloaders,use_gpu):
        if use_gpu:
            features = torch.FloatTensor()
        count = 0
        if opt.linear_num <= 0:
            opt.linear_num = 2048

        for iter, data in enumerate(dataloaders):
            print("len(dataloaders):",len(dataloaders),'iter:',iter)
            img, label = data
            n, c, h, w = img.size()
            count += n
            print('label-dataloader:',label)
            #print(count)
            # ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()
            # ff = torch.FloatTensor(n,opt.linear_num).zero_().cpu()
            ff = torch.FloatTensor(n,opt.linear_num).zero_().to(device)
            

            for i in range(2):
                if(i==1):
                    img = fliplr(img)
                # input_img = Variable(img.cuda())
                # input_img = Variable(img.cpu())
                input_img = img.to(device)
                for scale in ms:
                    if scale != 1:
                        # bicubic is only  available in pytorch>= 1.1
                        input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                    outputs = model(input_img) 
                    ff += outputs
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            
            if iter == 0:
                features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
            start = iter*opt.batchsize
            end = min( (iter+1)*opt.batchsize, len(dataloaders.dataset))
            features[ start:end, :] = ff
        return features

    def get_id(img_path):
        camera_id = []
        labels = []
        name_ids =[]
        frame_ids =[]
        paths = []
        for path, v in img_path:
            #filename = path.split('/')[-1]
            name = os.path.basename(path)
            name1= name[0:-4]
            ID  = name1.split('_')
            name_id = name1
            label = ID[2]
            camera = ID[0]
            frame = ID[1][1:]
            labels.append((label))
            camera_id.append((camera))
            name_ids.append(name_id)
            frame_ids.append(int(frame))
            paths.append(path)
        return camera_id, labels, name_ids, frame_ids ,paths

    gallery_path = image_datasets[gallery].imgs
    query_path = image_datasets[query].imgs

    gallery_cam,gallery_label ,gallery_name, gallery_frame,gallery_path = get_id(gallery_path)
    query_cam,query_label ,query_name ,query_frame,query_path = get_id(query_path)


    ######################################################################
    # Load Collected data Trained model
    print('-------test-----------')
    model_structure = ft_net(opt.nclasses, stride = opt.stride, ibn = opt.ibn, linear_num=opt.linear_num)
    model = load_network(model_structure)
    model.classifier.classifier = nn.Sequential()
    model = model.eval()
    model = model.to(device)
    # if use_gpu:
    #     print(use_gpu)
    #     model = model.to('cuda')
    #     # model = model.cuda()


    print('Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.')
    model = fuse_all_conv_bn(model)
    # dummy_forward_input = torch.rand(opt.batchsize, 3, h, w).cuda()
    # dummy_forward_input = torch.rand(opt.batchsize, 3, h, w).cpu()
    dummy_forward_input = torch.rand(opt.batchsize, 3, h, w).to(device)
    model = torch.jit.trace(model, dummy_forward_input)
    # Extract feature
    since = time.time()
    with torch.no_grad():
        gallery_feature = extract_feature(model,dataloaders[gallery],use_gpu)
        query_feature = extract_feature(model,dataloaders[query],use_gpu)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.2f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

    result = {'gallery_frame':gallery_frame,'query_frame':query_frame,'gallery_path':gallery_path,'query_path':query_path,'gallery_name':gallery_name,'query_name':query_name,'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam,'gallery_path':gallery_path,'query_path':query_path}
    if not os.path.exists("./mat_result"):
         os.makedirs("./mat_result")
    scipy.io.savemat('./mat_result/'+os.path.basename(gallery)+'_'+ os.path.basename(query)+'_pytorch_result.mat',result)
    # scipy.io.savemat('pytorch_image_datasets.mat',image_datasets)
    print("finish!")
    # result = './model/%s/result.txt'%opt.name
    #os.system('python evaluate_gpu.py | tee -a %s'%result)
