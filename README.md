# CIT: Complementary Interaction Transformer for Lightweight Super Resolution


> **Abstract:** *The lightweight super resolution methods based on deep learning have gained widespread attention in this field due to their better universality and applicability. Although Dual Aggregation Transformer (DAT) has achieved good results in lightweight super resolution, the DAT focuses too much on the transformer branch and ignores the parallel pointwise convolution branch in order to achieve lightweighting. Previous methods have lacked attention to the interaction of high-frequency and low-frequency information in the restoration of images. We believe that these two branches can maintain mutual interaction while complementing each other. In this paper, we propose a complementary interaction transformer (CIT) for lightweight super resolution, which enhances pointwise convolution parallel modules of the traditional DAT. This network introduces three crafted modules: high frequency attention block (HFAB), efficient integration block (EIB), and gated high frequency feedforward (GHFF). The HFAB and EIB combine to form our transformer complementary interaction transformer block (CITB). CITB integrates low frequency information in the image and HFAB precisely utilizes high frequency information. EIB enhances the cognitive dimension of the network by strengthening non-linearity and increasing the receptive field. Utimately, a backbone network structure is formed where high-low frequency information complement and interaction spatial-channel wise information. Each module is designed with consideration for its impact on number of parameters. Experimental results reveal that our complementary interaction transformer for lightweight super resolution that achieves the better super resolution results with a less number of parameters.* 


## Dependencies

- Python 3.8
- PyTorch 1.8.0
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)


## Contents

1. [Datasets](#Datasets)
1. [Models](#Models)
1. [Training](#Training)
1. [Testing](#Testing)
1. [Results](#Results)
1. [Citation](#Citation)
1. [Acknowledgements](#Acknowledgements)

---

## Datasets

Used training and testing sets can be downloaded as follows:

| Training Set                                                 |                         Testing Set                          |                        Visual Results                        |
| :----------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images, 100 validation images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) [complete training dataset DF2K: [Google Drive](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link) / [Baidu Disk](https://pan.baidu.com/s/1KIcPNz3qDsGSM0uDKl4DRw?pwd=74yc)] | Set5 + Set14 + BSD100 + Urban100 + Manga109 [complete testing dataset: [Google Drive](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1Tf8WT14vhlA49TO2lz3Y1Q?pwd=8xen)] | [Google Drive](https://drive.google.com/drive/folders/1ZMaZyCer44ZX6tdcDmjIrc_hSsKoMKg2?usp=drive_link) / [Baidu Disk](https://pan.baidu.com/s/1LO-INqy40F5T_coAJsl5qw?pwd=dqnv#list/path=%2F) |

Download training and testing datasets and put them into the corresponding folders of `datasets/`. See [datasets](datasets/README.md) for the detail of the directory structure.


## Training

- Download [training](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link) (DF2K, already processed) and [testing](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) (Set5, Set14, BSD100, Urban100, Manga109, already processed) datasets, place them in `datasets/`.

- Run the following scripts. The training configuration is in `options/train/`.

  ```shell
  # DAT-S, input=64x64, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_S_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_S_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_S_x4.yml --launcher pytorch
  
  # DAT, input=64x64, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_x4.yml --launcher pytorch
  
  # DAT-2, input=64x64, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_2_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_2_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_2_x4.yml --launcher pytorch
  
  # DAT-light, input=64x64, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_light_x2.yml --launcher pytorch
  PYTHONPATH="./:${PYTHONPATH}" python -m torch.distributed.launch --nproc_per_node=1 basicsr/train.py -opt options/Train/train_DAT_light_x2.yml --launcher pytorch --auto_resume

  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_light_x3.yml --launcher pytorch
  PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/Train/train_DAT_light_x4.yml --launcher pytorch
  PYTHONPATH="./:${PYTHONPATH}" python -m torch.distributed.launch --nproc_per_node=1 --master_port=1 basicsr/train.py -opt options/Train/train_DAT_light_x4.yml --launcher pytorch --auto_resume
  PYTHONPATH="./:${PYTHONPATH}" python -m torch.distributed.launch --nproc_per_node=1 basicsr/train.py -opt options/Train/train_DAT_light_x4.yml --launcher pytorch --auto_resume

  
  
  ```

- The training experiment is in `experiments/`.

## Testing

### Test images with HR

- Download the pre-trained [models](https://drive.google.com/drive/folders/1iBdf_-LVZuz_PAbFtuxSKd_11RL1YKxM?usp=drive_link) and place them in `experiments/pretrained_models/`.

  We provide pre-trained models for image SR: DAT-S, DAT, DAT-2, and DAT-light (x2, x3, x4).

- Download [testing](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) (Set5, Set14, BSD100, Urban100, Manga109) datasets, place them in `datasets/`.

- Run the following scripts. The testing configuration is in `options/test/` (e.g., [test_DAT_x2.yml](options/Test/test_DAT_x2.yml)).

  Note 1:  You can set `use_chop: True` (default: False) in YML to chop the image for testing.

  ```shell
  # No self-ensemble
  # DAT-S, reproduces results in Table 2 of the main paper
  python basicsr/test.py -opt options/Test/test_DAT_S_x2.yml
  python basicsr/test.py -opt options/Test/test_DAT_S_x3.yml
  python basicsr/test.py -opt options/Test/test_DAT_S_x4.yml
  
  # DAT, reproduces results in Table 2 of the main paper
  python basicsr/test.py -opt options/Test/test_DAT_x2.yml
  python basicsr/test.py -opt options/Test/test_DAT_x3.yml
  python basicsr/test.py -opt options/Test/test_DAT_x4.yml
  
  # DAT-2, reproduces results in Table 1 of the supplementary material
  python basicsr/test.py -opt options/Test/test_DAT_2_x2.yml
  python basicsr/test.py -opt options/Test/test_DAT_2_x3.yml
  python basicsr/test.py -opt options/Test/test_DAT_2_x4.yml
  
  # DAT-light, reproduces results in Table 2 of the supplementary material
  python basicsr/test.py -opt options/Test/test_DAT_light_x2.yml
  python basicsr/test.py -opt options/Test/test_DAT_light_x3.yml
  python basicsr/test.py -opt options/Test/test_DAT_light_x4.yml
  ```

- The output is in `results/`.

### Test images without HR

- Download the pre-trained [models](https://drive.google.com/drive/folders/1iBdf_-LVZuz_PAbFtuxSKd_11RL1YKxM?usp=drive_link) and place them in `experiments/pretrained_models/`.

  We provide pre-trained models for image SR: DAT-S, DAT, and DAT-2 (x2, x3, x4).

- Put your dataset (single LR images) in `datasets/single`. Some test images are in this folder.

- Run the following scripts. The testing configuration is in `options/test/` (e.g., [test_single_x2.yml](options/Test/test_single_x2.yml)).

  Note 1: The default model is DAT. You can use other models like DAT-S by modifying the YML.

  Note 2:  You can set `use_chop: True` (default: False) in YML to chop the image for testing.

  ```shell
  # Test on your dataset
  python basicsr/test.py -opt options/Test/test_single_x2.yml
  python basicsr/test.py -opt options/Test/test_single_x3.yml
  python basicsr/test.py -opt options/Test/test_single_x4.yml
  ```

- The output is in `results/`.

## Results

We achieved state-of-the-art performance. Detailed results can be found in the paper. All visual results of DAT can be downloaded [here](https://drive.google.com/drive/folders/1ZMaZyCer44ZX6tdcDmjIrc_hSsKoMKg2?usp=drive_link).

<details>
<summary>Click to expand</summary>

- results in Table 2 of the main paper

<p align="center">
  <img width="900" src="figs/Table-1.png">
</p>


- results in Table 1 of the supplementary material

<p align="center">
  <img width="900" src="figs/Table-2.png">
</p>


- results in Table 2 of the supplementary material

<p align="center">
  <img width="900" src="figs/Table-3.png">
</p>




- visual comparison (x4) in the main paper

<p align="center">
  <img width="900" src="figs/Figure-1.png">
</p>


- visual comparison (x4) in the supplementary material

<p align="center">
  <img width="900" src="figs/Figure-2.png">
  <img width="900" src="figs/Figure-3.png">
  <img width="900" src="figs/Figure-4.png">
  <img width="900" src="figs/Figure-5.png">
</p>
</details>



## Acknowledgements

This code is built on  [BasicSR](https://github.com/XPixelGroup/BasicSR).
