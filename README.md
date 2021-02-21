# About Repo FasterRCNN-FPN_Classification
This project is mainly about supervised learning and classification.
## Divide data set
```bash
python3 divideDataSet.py
./split.sh
```
## Train
If report a error `ModuleNotFoundError: No module named 'pycocotools'`
Please run the following command :
```bash
pip3 install cython
pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
And then run:
```bash
python3 train.py
```
## Test
```bash
python3 test.py
```
## Requires packages
torch--1.7.1
numpy-1.17.0
pycocotools-2.0
torchvision-0.8.2