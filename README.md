# AMHDM-SRGAN

[![DOI](https://zenodo.org/badge/934706872.svg)](https://doi.org/10.5281/zenodo.14908776)

## Citation
If you have used this code or data in your research, please cite the following papers:
```bibtex
@article{
  title    = {Hybrid-Order Adaptive and Multi-modal Degradation Model for Enhanced Blind Image Super-resolution},  
  author   = {Zhang, Xin and Yi, Huawei and Zhao, Mengyuan and Wang, Yajun and Lan, Jie},
  journal  = {The Visual Computer},
  year     = {2024},
}
Zhang, X., Yi, H., Zhao, M. et al. Hybrid-Order Adaptive and Multi-modal Degradation Model for Enhanced Blind Image Super-resolution. Vis Comput(2024).
```

## How to Train/Finetune AMHDM-SRGAN

- [Train AMHDM-SRGAN](#train-amhdm-srgan)
  - [Overview](#overview)
  - [Dataset Preparation](#dataset-preparation)
  - [Train AMHDM-SRNet](#Train-AMHDM-SRNet)
  - [Train AMHDM-SRGAN](#Train-AMHDM-SRGAN)


### Environmental dependencies
- Python 3.8+ / PyTorch 1.9+ / CUDA 11.1
 Install commands:'pip install -r requirements.txt'

### Train AMHDM-SRGAN

#### Overview

The training has been divided into two stages. These two stages have the same data synthesis process and training pipeline, except for the loss functions. Specifically,

1. We first train AMHDM-SRNet with L1 loss from the pre-trained model ESRGAN.
1. We then use the trained AMHDM-SRNet model as an initialization of the generator, and train the AMHDM-SRGAN with a combination of L1 loss, perceptual loss and GAN loss.

#### Dataset Preparation

We use DIV2K  datasets for our training. Only HR images are required. <br>
You can download from :

DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
Flickr2K: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
3Set5: https://github.com/jbhuang0604/SelfExSR
Set14: https://github.com/jbhuang0604/SelfExSR
BSD100: https://github.com/jbhuang0604/SelfExSR
Urban100: https://github.com/jbhuang0604/SelfExSR
RealSR:https://drive.google.com/file/d/1gKnm9BdgyqISCTDAbGbpVitT-QII_unw/view?usp=drive_open
DRealSR:https://drive.google.com/drive/folders/1tP5m4k1_shFT6Dcw31XV8cWHtblGmbOk
Here are steps for data preparation.

#### Step 1: [Optional] Generate multi-scale images

For the DIV2K dataset, we use a multi-scale strategy, *i.e.*, we downsample HR images to obtain several Ground-Truth images with different scales. <br>
You can use the [scripts/generate_multiscale.py](scripts/generate_multiscale_DF2K.py) script to generate multi-scale images. <br>
Note that this step can be omitted if you just want to have a fast try.

```bash
python scripts/generate_multiscale.py --input datasets/DF2K_train_HR --output datasets/DF2K_train_HR_multiscale
```

#### Step 2: [Optional] Crop to sub-images

We then crop DF2K images into sub-images for faster IO and processing.<br>
This step is optional if your IO is enough or your disk space is limited.

You can use the [scripts/extract_subimages.py](scripts/extract_subimages.py) script. Here is the example:

```bash
 python scripts/extract_subimages.py --input datasets/DF2K_train_HR_multiscale --output datasets/DF2K_train_HR_multiscale_sub --crop_size 400 --step 200
```

#### Step 3: Prepare a txt for meta information

You need to prepare a txt file containing the image paths. The following are some examples in `meta_info_DF2K_train_HR_multiscale_sub.txt` (As different users may have different sub-images partitions, this file is not suitable for your purpose and you need to prepare your own txt file):

```txt
DF2K_train_HR_multiscale_sub/000001_s001.png
DF2K_HR_sub/000001_s002.png
DF2K_HR_sub/000001_s003.png
...
```

You can use the [scripts/generate_meta_info.py](scripts/generate_meta_info.py) script to generate the txt file. <br>
You can merge several folders into one meta_info txt. Here is the example:

```bash
 python scripts/generate_meta_info.py --input datasets/DIV2K_train_HR datasets/DIV2K_train_HR_multiscale --root datasets/DIV2K_train_HR datasets/DIV2K_train_HR --meta_info datasets/DIV2K_train_HR/meta_info/meta_info_DIV2K_train_HRmultiscale.txt
```

### Train AMHDM-SRNet

1. Download pre-trained model [ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth) into `experiments/pretrained_models`.
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth -P experiments/pretrained_models
    ```
1. Modify the content in the option file `options/train_amhdmsrnet_x4plus.yml` accordingly:
    ```yml
    train:
        name: DF2K
        type: AMHDMSRGANDataset
        dataroot_gt: datasets/DF2K_train_HR  # modify to the root path of your folder
        meta_info: amhdmsrgan/meta_info/meta_info_DF2K_train_HR_sub.txt  # modify to your own generate meta info txt
        pretrain_network_g: experiments/pretrained_models/ESRGAN_SRx4.pth
        io_backend:
            type: disk
    ```
1. If you want to perform validation during training, uncomment those lines and modify accordingly:
    ```yml
      # Uncomment these for validation
      # val:
      #   name: validation
      #   type: PairedImageDataset
      #   dataroot_gt: path_to_gt
      #   dataroot_lq: path_to_lq
      #   io_backend:
      #     type: disk

    ...

      # Uncomment these for validation
      # validation settings
      # val:
      #   val_freq: !!float 5e3
      #   save_img: True

      #   metrics:
      #     psnr: # metric name, can be arbitrary
      #       type: calculate_psnr
      #       crop_border: 4
      #       test_y_channel: false
    ```

 

    Train with **a single GPU**:
    ```bash
    python amhdmsrgan/train.py -opt options/train_amhdmsrnet_x4plus.yml --auto_resume
    ```

### Train AMHDM-SRGAN

1. After the training of AMHDM-SRNet, you now have the file `experiments\pretrained_models\net_d_160000.pth`. If you need to specify the pre-trained path to other files, modify the `pretrain_network_g` value in the option file `train_amhdmsrgan_x4plus.yml`.
1. Modify the option file `train_amhdmsrgan_x4plus.yml` accordingly. Most modifications are similar to those listed above.
1. Before the formal training, you may run in the `--debug` mode to see whether everything is OK. We use four GPUs for training:


    Train with **a single GPU**:
    ```bash
    python amhdmsrgan/train.py -opt options/train_amhdmsrgan_x4plus.yml --auto_resume
    ```
### Run AMHDM-SRGAN
1. Download pre-trained model [AMHDM-SRGAN](https://zenodo.org/records/14908871) into `experiments/pretrained_models`.
    ```bash
    wget https://zenodo.org/records/14908871/net_g_380000.pth -P experiments/pretrained_models
    ```
    ```
    python amhdmsrgan.py
    ```
