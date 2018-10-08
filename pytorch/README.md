# Pytorch Implementation of Region Ensemble Network

[[arxiv]](https://arxiv.org/abs/1702.02447) [[offical caffe code]](https://github.com/guohengkai/region-ensemble-network)

## Notes

Pull requests are welcome.

I did not replicate the error results as that was not the goal of my project.  
Results during the demo are pretty good.

I also added another option to change the input size. The network will splice the features according. It will give an assertion error if the input size results in the features dimension being odd number.

## Installing

`pytorch 0.4.0`  
`opencv 3.4.0`  


## Data
Right now the code only supports the MSRA Hand Gesture dataset. If I have time, i will add support for IVCL and NYU.

Put the MSRA dataset in data/


    data/
    data/P0/..
    data/P1/1/000000.bin
    ... and several other similar folders...


## Training

It is assumed that CUDA is installed.  

`python main.py --name experiment_1 --save_dir experiments/ --batchSize 128`

To resume training, add the checkpoint option accordingly.

`python main.py --checkpoint checkpoint.pth.tar`

To use a model as a pretrained checkpoint/finetuning,

`python main.py --checkpoint checkpoint.pth.tar --finetune`

If you only want to train on certain poses/subjects,

`python main.py --poses 1 2 3 --persons 1 2 3`



## Getting the training statistics

The train loss and validation loss will be saved in an out file in `save_dir`

## Acknowledgments

Many thanks to the original authors of the paper who readily and kindly offered help
