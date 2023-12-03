# CARGNet
We propose a novel weakly supervised change detection framework named CARGNet for point supervised change detection. The overall framework for CARGNet is as follows:
![markdown](./images/framework.jpg)
## Requirements
```
- python 3.8
- pytorch 1.7.1
- opencv-python 4.8.1.78
- torchvision 0.8.2
- pillow 9.0.1
- tqdm 4.63.0
```
## Getting Started
### Data Download
The LEVIR-CD-Point dataset can be downloaded from: [https://pan.baidu.com/s/1bV1TCNxbloJveqh1eG3a7w?pwd=dskl](https://pan.baidu.com/s/1bV1TCNxbloJveqh1eG3a7w?pwd=dskl) 

The DSIFN-CD-Point dataset can be downloaded from: [https://pan.baidu.com/s/12wkHXxStmlrgcNk3yMdqyA?pwd=dlst](https://pan.baidu.com/s/12wkHXxStmlrgcNk3yMdqyA?pwd=dlst) 

Then put LEVIR-CD-Point and DSIFN-CD-Point datasets into datasets folder.
## Evaluate
### 1. Download our [weights](https://pan.baidu.com/s/1RkEPaV-hGVjVn0eSQ3Dbqw?pwd=xthc)
### 2. Run our code
```
python test.py
```
## Train our network 
```
python train.py
```
## Dataset Structure
* `LEVIR-CD-Point`:
    * `train`:
      * `C`：Changed images.
      * `UC`：Unchanged images.
    * `test`:
      * `image`
      * `image2`
      * `label`
* `DSIFN-CD-Point`:
    * `train`:
      * `C`：Changed images.
      * `UC`：Unchanged images.
    * `test`:
      * `image`
      * `image2`
      * `label`
