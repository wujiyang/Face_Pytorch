## Face_Pytorch
The implementation of  popular face recognition algorithms in pytorch framework, including arcface, cosface and sphereface and so on.

All codes are evaluated on Pytorch 0.4.0 with Python 3.6, Ubuntu 16.04.10, CUDA 9.1 and CUDNN 7.1.


### Train with CASIA-WebFace

  Model Type |   Loss    | LFW Acc. | AgeDB30 |MF Acc.@R|MF Ver.@R | SIZE 
:-----------:|:---------:|:--------:|:-------:|:-------:|:--------:|:-----:
MobileFaceNet|  ArcFace  |  0.9922  |  0.9257 | 0.7645  |  0.8195  |  4MB
LResNet-50   |  ArcFace  |          |         |         |          | 292MB 
LResNet-101  |  ArcFace  |          |         |         |          |


### Train with MS-Celeb-1M

 Model Type |   Loss    | LFW Acc. |AgeDB30|MF Acc.@R|MF Ver.@R | SIZE 
:-----------:|:---------:|:--------:|:------:|:-------:|:--------:|:-----:
MobileFaceNet|  ArcFace  |  0.9922  | 0.9257 | 0.7645  |  0.8195  |  4MB
LResNet-50   |  ArcFace  |          |        |         |          | 292MB 
LResNet-101  |  ArcFace  |          |        |         |          |


### Train with VGGFace2

 Model Type |   Loss    | LFW Acc. |AgeDB30|MF Acc.@R|MF Ver.@R | SIZE 
:-----------:|:---------:|:--------:|:------:|:-------:|:--------:|:-----:
MobileFaceNet|  ArcFace  |  0.9922  | 0.9257 | 0.7645  |  0.8195  |  4MB
LResNet-50   |  ArcFace  |          |        |         |          | 292MB 
LResNet-101  |  ArcFace  |          |        |         |          |




### References
[CosFace_pytorch](https://github.com/MuggleWang/CosFace_pytorch)  
[MobileFaceNet_Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)  
[InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)