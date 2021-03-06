#coding:utf-8
import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import datasets, transforms#transforms只能对图片进行增强，无法同时改变图片对应的标签，故使用torchvision Github项目中的函数
import utils
import transforms as T
from engine import train_one_epoch, evaluate
from PIL import Image
from xml.dom.minidom import parse
#%matplotlib inline
#--------------------
#定义数据集
class MarkDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
 
    def __getitem__(self, idx):
        # load images and bbox
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        bbox_xml_path = os.path.join(self.root, "Annotations", self.bbox_xml[idx])
        img = Image.open(img_path).convert("RGB")        
        
        # 读取文件，VOC格式的数据集的标注是xml格式的文件
        dom = parse(bbox_xml_path)
        # 获取文档元素对象
        data = dom.documentElement
        # 获取 objects
        objects = data.getElementsByTagName('object')        
        # get bounding box coordinates
        boxes = []
        labels = []
        for object_ in objects:
            # 获取标签中内容
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue  # 就是label，cat_1或dog_2
            labels.append(np.int(name.split('_')[-1]))  # 背景的label是0，cat_1和dog_2的label分别是1和2
            
            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])        

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)        
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # 由于训练的是目标检测网络，因此没有教程中的target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            # 注意这里target(包括bbox)也转换\增强了，和from torchvision import的transforms的不同
            # https://github.com/pytorch/vision/tree/master/references/detection 的 transforms.py里就有RandomHorizontalFlip时target变换的示例
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    #if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # 50%的概率水平翻转
        # transforms.append(T.RandomHorizontalFlip(0.5))
 
    return T.Compose(transforms)
#---------------------------------------------------
root = './data'

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 3 classes, mark_type_1，mark_type_2，background
num_classes = 3
# use our dataset and defined transformations
dataset = MarkDataset(root, get_transform(train=True))
dataset_test = MarkDataset(root, get_transform(train=False))

# split the dataset in train and test set
# 我的数据集一共有20张图，差不多训练验证1:1
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:8])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-8:])


# define training and validation data loaders
# 在jupyter notebook里训练模型时num_workers参数只能为0，不然会报错，这里就把它注释掉了
data_loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=2,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, 
    batch_size=1, 
    shuffle=False, 
    num_workers=2,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)  # 或get_object_detection_model(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

# SGD
optimizer = torch.optim.SGD(params, lr=0.003, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
# cos学习率
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

# let's train it for   epochs
num_epochs = 10
resume = True

for epoch in range(1, num_epochs+1):
    # train for one epoch, printing every 10 iterations
    # engine.py的train_one_epoch函数将images和targets都.to(device)了
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)

    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset    
    evaluate(model, data_loader_test, device=device)    
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    checkpoint = {
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(checkpoint, 'checkpoint/weight.ckpt')

#    if resume:
#        print("Resume from checkpoint...")
#        checkpoint = torch.load(CHECKPOINT_FILE)
#        model.load_state_dict(checkpoint['model_state_dict'])
#        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#        epoch = checkpoint['epoch']
#        model.eval()
    torch.save(model, './model/weight_epoch-{}.pt'.format(epoch))
