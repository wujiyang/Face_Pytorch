## Face_Pytorch
The implementation of  popular face recognition algorithms in pytorch framework, including arcface, cosface and sphereface and so on.

All codes are evaluated on Pytorch 0.4.0 with Python 3.6, Ubuntu 16.04.10, CUDA 9.1 and CUDNN 7.1.


## Data Preparation
For CNN training, I use CASIA-WebFace and Cleaned MS-Celeb-1M, aligned by MTCNN with the size of 112x112.  For performance testing, I report the results on LFW, AgeDB30, MegaFace Identification and Verification.

For AgeDB-30 and CFP-FP, the aligned images and evaluation pairs are restored from the mxnet binary file provided by [insightface](https://github.com/deepinsight/insightface), tools are available in this repository. You should install a mxnet-cpu first for the image parsing, just do ' **pip install mxnet** ' is ok.

## Results
> MobileFaceNet: Struture described in MobileFaceNet  
> ResNet50: Original resnet structure  
> ResNet50-IR: CNN described in ArcFace paper  
> SEResNet50-IR: CNN described in ArcFace paper 
### Verification result on LFW, AgeDB-30 and CFP_FP
Train on CASIA-WebFace (small protocol)

  Model Type    |   Loss    | LFW Acc. | AgeDB-30 Acc.| CFP-FP Acc. |  SIZE 
:--------------:|:---------:|:--------:|:------------:|:-----------:|:------:|
MobileFaceNet   |  ArcFace  |  0.9922  |    0.9257    |   0.9310    |  4MB
ResNet50        |  ArcFace  |          |              |             | 292MB 
ResNet50-IR     |  ArcFace  |          |              |             |         
SEResNet50-IR   |  ArcFace  |          |              |             |         

Train on MS-Celeb-1M (large protocol) 

  Model Type    |   Loss    | LFW Acc. | AgeDB-30 Acc.| CFP-FP Acc. |  SIZE 
:--------------:|:---------:|:--------:|:------------:|:-----------:|:------:|
MobileFaceNet   |  ArcFace  |          |              |             |  4MB
ResNet50        |  ArcFace  |          |              |             | 292MB 
ResNet50-IR     |  ArcFace  |          |              |             |         
SEResNet50-IR   |  ArcFace  |          |              |             |        

### MegaFace Rank 1 Identifiaction and Verfication with TPR@FPR=1e-6
Train on CASIA-WebFace (small protocol) 

  Model Type    |   Loss    | MF Acc. | MF Ver. | MF Acc.@R | MF Ver.@R |  SIZE 
:--------------:|:---------:|:-------:|:-------:|:---------:|:---------:|:-----:
MobileFaceNet   |  ArcFace  |         |         |           |           |  4MB
ResNet50        |  ArcFace  |         |         |           |           | 292MB 
ResNet50-IR     |  ArcFace  |         |         |           |           |
SEResNet50-IR   |  ArcFace  |         |         |           |           |

Train on MS-Celeb-1M (large protocol)

  Model Type    |   Loss    | MF Acc. | MF Ver. | MF Acc.@R | MF Ver.@R |  SIZE 
:--------------:|:---------:|:-------:|:-------:|:---------:|:---------:|:-----:
MobileFaceNet   |  ArcFace  |         |         |           |           |  4MB
ResNet50        |  ArcFace  |         |         |           |           | 292MB 
ResNet50-IR     |  ArcFace  |         |         |           |           |
SEResNet50-IR   |  ArcFace  |         |         |           |           |


### References
[CosFace_pytorch](https://github.com/MuggleWang/CosFace_pytorch)  
[MobileFaceNet_Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)  
[InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)  
[insightface](https://github.com/deepinsight/insightface)