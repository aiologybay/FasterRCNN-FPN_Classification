import os
import random 

def segmentation_xml_dataset(xmlfilepath,saveBasePath):
	trainval_percent=0.8
	train_percent=0.25
	total_xml = os.listdir(xmlfilepath)
	num=len(total_xml)  
	list=range(num)  
	tv=int(num*trainval_percent)  
	tr=int(tv*train_percent)  
	trainval= random.sample(list,tv)  
	train=random.sample(trainval,tr)  
	print("train size:",tr)
	print("train and val size:",tv)

	ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
	ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
	ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
	fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
	 
	for i in list:
		name=total_xml[i][:-4]+'\n'  
		if i in trainval:  
			ftrainval.write(name)  
			if i in train:
				ftrain.write(name)  
			else:  
				fval.write(name)  
		else:
			ftest.write(name)  
	  
	ftrainval.close()  
	ftrain.close()  
	fval.close()  
	ftest.close()

def segmentation_jpg_dataset(imgfilepath, saveBasePath):
	trainval_percent=0.8
	train_percent=0.5
	total_img = os.listdir(imgfilepath)
	num=len(total_img)  
	list=range(num)  
	tv=int(num*trainval_percent) #trainval num
	tr=int(tv*train_percent)  #train num
	te=int(num-tv)
	trainval = random.sample(list,tv)
	train = random.sample(trainval,tr)  
	print("train size:",tr)
	print("val size:",tv-tr)
	print("test size:",te)

	ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
	ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
	ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
	fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
	 
	for i in list:
		name=os.path.join(imgfilepath,total_img[i]) + '\n'
		if i in trainval:  
			ftrainval.write(name)  
			if i in train:
				ftrain.write(name)  
			else:  
				fval.write(name)  
		else:
			ftest.write(name)  

	ftrainval.close()  
	ftrain.close()  
	fval.close()  
	ftest.close()
if __name__=='__main__':
	jpgfilepath="./data/JPEGImages"
	saveBasePath="./data/Main"
	segmentation_jpg_dataset(jpgfilepath,saveBasePath)
