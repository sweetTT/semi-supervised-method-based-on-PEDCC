

# Requirements
Tensorflow 1.13  
Python 2.7
# Results

 Data| CIFAR10(4K)| SVHN(1K)
 ---- | ----- | ------  
 Test error rate(%)| 4.90| 2.42
# Method
<img src="https://github.com/sweetTT/semi-supervised-method-based-on-PEDCC/blob/master/images/figure.png" width="512">

# Preprocess
```
bash scripts/preprocess_cifar10.sh
bash scripts/preprocess_svhn.sh
```  

# Run on GPU
```
bash scripts/cifar10_400_0.2.sh
bash scripts/svhn_gpu_1600_0.04
```  
# Citation
Please cite this paper if you think it is useful for you.  
Title: Semi-supervised learning method based on predefined evenly-distributed class centroids  
Author: Qiuyu Zhu, Tiantian Li
