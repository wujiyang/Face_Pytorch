## Face_Pytorch
The implementation of  popular face recognition algorithms in pytorch framework, including arcface, cosface and sphereface and so on.

All codes are evaluated on Pytorch 0.4.0 with Python 3.6, Ubuntu 16.04.10, CUDA 9.1 and CUDNN 7.1.


## Data Preparation
For CNN training, I use CASIA-WebFace and Cleaned MS-Celeb-1M, aligned by MTCNN with the size of 112x112.  For performance testing, I report the results on LFW, AgeDB30, MegaFace Identification and Verification.

For AgeDB-30 and CFP-FP, the aligned images and evaluation pairs are restored from the mxnet binary file provided by [insightface](https://github.com/deepinsight/insightface), tools are available in this repository. You should install a mxnet-cpu first for the image parsing, just do ' **pip install mxnet** ' is ok.  
[LFW @ BaiduNetdisk](https://pan.baidu.com/s/1Rue4FBmGvdGMPkyy2ZqcdQ),   [AgeDB-30 @ BaiduNetdisk](https://pan.baidu.com/s/1sdw1lO5JfP6Ja99O7zprUg),   [CFP_FP @ BaiduNetdisk](https://pan.baidu.com/s/1gyFAAy427weUd2G-ozMgEg)

## Results
> MobileFaceNet: Struture described in MobileFaceNet  
> ResNet50: Original resnet structure  
> ResNet50-IR: CNN described in ArcFace paper  
> SEResNet50-IR: CNN described in ArcFace paper 
### Verification result on LFW, AgeDB-30 and CFP_FP  
Small Protocol: trained with CASIA-WebFace of size: 453580/10575  
Large Protocol: trained with Cleaned MS-Celeb-1M of size: 3923399/86876

  Model Type    |   Loss    | LFW Acc. | AgeDB-30 Acc.| CFP-FP Acc. |  SIZE  | protocol
:--------------:|:---------:|:--------:|:------------:|:-----------:|:------:|:--------:
MobileFaceNet   |  ArcFace  |  0.9923  |    0.9326    |   0.9434    |  4MB   |  small
ResNet50-IR     |  ArcFace  |  0.9942  |    0.9445    |   0.9534    | 170MB  |  small  
SEResNet50-IR   |  ArcFace  |          |              |             |        |  small
MobileFaceNet   |  ArcFace  |          |              |             |  4MB   |  large
ResNet50-IR     |  ArcFace  |          |              |             | 170MB  |  large
SEResNet50-IR   |  ArcFace  |          |              |             |        |  large

### MegaFace Rank 1 Identifiaction and Verfication with TPR@FPR=1e-6

  Model Type    |   Loss    | MF Acc. | MF Ver. | MF Acc.@R | MF Ver.@R |  SIZE | protocol
:--------------:|:---------:|:-------:|:-------:|:---------:|:---------:|:-----:|:-------:
MobileFaceNet   |  ArcFace  | 0.6910  | 0.8423  |  0.8115   |  0.8586   |  4MB  |  small
ResNet50-IR     |  ArcFace  | 0.7431  | 0.8823  |  0.8744   |  0.8956   | 170MB |  small
SEResNet50-IR   |  ArcFace  |         |         |           |           |       |  small
MobileFaceNet   |  ArcFace  |         |         |           |           |  4MB  |  large
ResNet50-IR     |  ArcFace  |         |         |           |           |       |  large
SEResNet50-IR   |  ArcFace  |         |         |           |           |       |  large


### References
[MuggleWang/CosFace_pytorch](https://github.com/MuggleWang/CosFace_pytorch)  
[Xiaoccer/MobileFaceNet_Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)  
[TreB1eN/InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)  
[deepinsight/insightface](https://github.com/deepinsight/insightface)