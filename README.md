# CARGNet
we propose a novel weakly supervised change detection framework named CARGNet for point supervised change detection. The overall framework for CARGNet is as follows:
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
## Dataset
### Dataset Structure
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
### Data Download
The LEVIR-CD-Point dataset can be downloaded from: [here](https://pan.baidu.com/s/1IeKRxOfuvyh0Q2LHIOy2iA) code：dskl

The DSIFN-CD-Point dataset can be downloaded from: [here](https://pan.baidu.com/s/1cI4w76yKG2C6GIPYKOcrIA) code：dlst
