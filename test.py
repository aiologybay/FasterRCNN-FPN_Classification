#coding:utf-8
import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import json

CLASSES=({'index':0, 'value': '__background__'},
	{'index':1, 'value': 'cat_1'},
	{'index':2, 'value': 'dog_2'},)


def showbbox(model, img, jpg_name):
    # 输入的img是0-1范围的tensor 

    model.eval()
    with torch.no_grad():
        '''
        prediction形如：
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
                           [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        prediction = model([img.to(device)])

    #print('prediction[0]=',prediction[0])
    img = img.permute(1,2,0)  # C,H,W → H,W,C，用来画图
    img = (img * 255).byte().data.cpu()  # * 255，float转0-255
    img = np.asarray(img)  # tensor → ndarray
    
    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin = round(prediction[0]['boxes'][i][0].item())
        ymin = round(prediction[0]['boxes'][i][1].item())
        xmax = round(prediction[0]['boxes'][i][2].item())
        ymax = round(prediction[0]['boxes'][i][3].item())
        label = prediction[0]['labels'][i].item()
        score = prediction[0]['scores'][i].item()

        if label == int(text[label]['index']):
            if score>=0.010:
                img = cv2.rectangle(img.copy(), (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
                if type(img) is cv2.UMat:
                    img=img.get()    #UMat to np.array
                img=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                draw=ImageDraw.Draw(img)
                draw.text((xmin,ymin-50), '预测:' + text[label]['value'] + '(图片:' + jpg_name + '_score:' + '%.2f%%' % (score * 100) + ')', font=ImageFont.truetype('NotoSansCJK-Regular.ttc', 20), fill=(255,0,0))
                img=cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
                img=np.asarray(img)
                with open('./class.txt', 'r') as f:

                    pred_cls=str(text[label]['value'].split('_')[0])
                    for line in f:
                        real_name=str(line.split()[0])
                        real_label=int(line.split('_')[-1])
                        real_cls=str(line.strip().split('.jpg ')[-1]).split('_')[0]

                        if jpg_name==real_name and label==real_label:
                            result='True'
                            print('{:<25}{:<34}{:<34}{:<12.3f}{:<5}'.format(jpg_name,real_cls, pred_cls, score,result))
                            res.write('{:<25}{:<34}{:<34}{:<12.3f}{:<5}\n'.format(jpg_name,real_cls, pred_cls, score,result))
                            break

                        elif jpg_name==real_name and label!=real_label:
                            result='False'
                            print('{:<25}{:<34}{:<34}{:<12.3f}{:<5}'.format(jpg_name,real_cls, pred_cls, score,result))
                            res.write('{:<25}{:<34}{:<34}{:<12.3f}{:<5}\n'.format(jpg_name,real_cls, pred_cls, score,result))
                            break
                	#print('预测结果为:{}'.format(text[label]['value'].split('_')[0]))
                #print('label:',text[label]['index'])
                	#print('scores:%.2f%%' % (score * 100))

                img=cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                cv2.namedWindow('picture', 0)
                cv2.resizeWindow('picture', 400, 400)
                cv2.imshow('picture', img)
                cv2.waitKey(10)
                cv2.imwrite(save_dir+'/'+jpg_name, img)
                break
				
    '''
    k = cv2.waitKey(2)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite(save_dir+jpg_name, img)
        cv2.destroyAllWindows()
    '''
#----------------------------------------------
if __name__ == '__main__':
    save_dir = './saved'
    path = './data/testjpg'
    res=open('./result.txt', 'w')
    model = torch.load(('./model/weight_epoch-3.pt'), map_location='cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    jpg_list=os.listdir(path)
    jpg_list.sort()
    f=json.dumps(CLASSES)
    text=json.loads(f)
    print('{:<25}{:<34}{:<34}{:<12}{:<5}'.format('jpg_name', 'real_cls', 'pred_cls', 'score', 'result'))
    res.write('{:<25}{:<34}{:<34}{:<12}{:<5}\n'.format('jpg_name', 'real_cls', 'pred_cls', 'score', 'result'))
    for jpg_name in jpg_list:
        transform1=transforms.Compose([
            transforms.ToTensor(),])
        img=Image.open(os.path.join(path, jpg_name)).convert('RGB')

        img=transform1(img)
        showbbox(model, img, jpg_name)
        #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
