# A Deeper Dive Into What Deep Spatiotemporal Networks Encode: Quantifying Static vs. Dynamic Information
Official Implementation of our CVPR 2022 Paper.

[pdf - coming soon!](), [project page](https://yorkucvil.github.io/Static-Dynamic-Interpretability/), [demo](https://youtu.be/H9bAqq-6-tg)


<div align="center">
<img src="https://github.com/YorkUCVIL/Static-Dynamic-Interpretability/blob/master/figures/teaser.png" width="50%" height="50%"><br><br>
</div>


## Description:

Deep spatiotemporal models are used in a variety of computer vision tasks, such as action recognition and video object segmentation. Currently, there is a limited understanding of what information is captured by these models in their intermediate representations. For example, while it has been observed that action recognition algorithms are heavily influenced by visual appearance in single static frames,
there is no quantitative methodology for evaluating such static bias in the latent representation compared to bias toward dynamic information (\eg motion).
We tackle this challenge by proposing a novel approach for quantifying the static and dynamic biases of any spatiotemporal model. To show the efficacy of our approach, we analyse two widely studied tasks, action recognition and video object segmentation.
Our key findings are threefold: (i) Most examined spatiotemporal models are biased toward static information; although, certain two-stream architectures with cross-connections show a better balance between the static and dynamic information captured. (ii) Some datasets that are commonly assumed to be biased toward dynamics are actually biased toward static information. (iii) Individual units (channels) in an architecture can be biased toward static, dynamic or a combination of the two.


<div align="center">
<img src="https://github.com/YorkUCVIL/Static-Dynamic-Interpretability/blob/master/figures/static_dynamic.png" width="100%" height="100%"><br><br>
</div>

## Installation
* Tested with Python3.7 and CUDA10.1
```
pip install -r requirements.txt
```

* Add fvcore + Pycoco Tools + Detectron2
```
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
```

## Data 

### Action Recognition Datasets

Use these links to download the following action recogintion datasets: 
[ActivityNet](http://activity-net.org/download.html)
[Diving48](http://www.svcl.ucsd.edu/projects/resound/dataset.html)
[SSv2](https://developer.qualcomm.com/software/ai-datasets/something-something)

### Video Stylization

We use the Interactive Video Stylization Using Few-Shot Patch-Based Training (https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training)
to stylize our videos. We simply run ```generate.py``` with the provided four [pretrained models.](https://drive.google.com/file/d/11_lCPqKDAtkMQTCSNTKBu2Sii8km_04s/view?usp=sharing)
For each dataset of interest, we generate four stylized versions of the validation set 
with the following data structure:

```
stylized_dataset
 |-- style_1 
 |   |-- video_1
 |      |-- frame_00001.jpg
 |      |-- frame_00002.jpg
 |      .
 |      .
 |      .
 |      |-- frame_00999.jpg
 |-- style_2 
 |   |-- video_1
 |      |-- frame_00001.jpg
 |      |-- frame_00002.jpg
 |      .
 |      .
 |      .
```

then set config.stylized_data_dir to point to the root directory.

## Quantifying static and dynamic neurons

### Action Recognition

Change the arguments in config.py, select the dataset
and the model out of the models listed in the get_model() function in the utils.py file. 

We obtain all pretrained models from the [SlowFast](https://github.com/facebookresearch/SlowFast) and
[TimeSformer](https://github.com/facebookresearch/TimeSformer) model zoos. We train our own models on Diving48
using modified config files from the SlowFast repository. 

Example of calculating the layerwise statistics:

```
python main.py --stylized_data_dir /path/to/stylized_dataset --dataset StylizedActivityNet --model i3d --stg 5
```

which quantifies the statics and dynamics for the i3d model on Stylized ActivityNet on the last
stage of the network (i.e., ResNet block).

To compute the joint-encoding statistics:

```
python main.py --joint_encoding True --stylized_data_dir /path/to/stylized_dataset --dataset StylizedActivityNet --model i3d --stg 5
```


### Video Object Segmentation

* Download Stylized DAVIS from [here](https://www.dropbox.com/s/we2c3m5d1f6qyol/Stylized_DAVIS.zip?dl=0)

* Download Weights for three VOS models from [here](https://www.dropbox.com/s/jhc3z21b0cbhb87/checkpoints_3models_staticdynamic_cvpr22.zip?dl=0)

* Run following command
```
bash run_main_vos.sh
```

* Statistics will be saved in dim_outputs/vos_models/final

* Compute layerwise statistics
```
python scripts/mean_var_analysis.py dim_outputs/vos_models/final/MODEL_NAME
```

* Compute unitwise statistics
```
python misc/plot_jointencoding.py dim_outputs/vos_models/joint_encoding/MODEL_NAME
```

* For training MATNet variants and the best performing model on MoCA dataset refer to this [repository](https://github.com/MSiam/MATNet_FusionCrossConStudy)

## Estimate Flops

### Action Recognition

We use [this function](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/utils/misc.py#L137)
from the SlowFast repository for all models.

### Video Object Segmentation
```
python misc/estimate_flops_params.py --checkpoint checkpoints_3models_staticdynamic_cvpr22/checkpoints_fseg/latest_twostream_deeplabv3plus_resnet101_davis_os16.pth --cfg_file configs/twostreamv3plus_davis.json --random_seed 1

python misc/estimate_flops_params.py --checkpoint checkpoints_3models_staticdynamic_cvpr22/checkpoints_matnet/ --cfg_file configs/matnet_davis.json --random_seed 1

python misc/estimate_flops_params.py --checkpoint checkpoints_3models_staticdynamic_cvpr22/model_RX50.pth --cfg_file configs/rtnet_davis.json --random_seed 1
```


## BibTeX
If you find this repository useful, please consider giving a star :star: and citation :t-rex:


      @InProceedings{kowal2022deeper,
       title={A Deeper Dive Into What Deep Spatiotemporal Networks Encode: Quantifying Static vs. Dynamic Information},
       author={Kowal, Matthew and Siam, Mennatullah and Islam, Md Amirul and Bruce, Neil and Wildes, Richard P. and Derpanis, Konstantinos G.},
       booktitle={Conference on Computer Vision and Pattern Recognition},
       year={2022}
     }

## References
* VOS: This repository heavily borrows from both [MATNet](https://github.com/tfzhou/MATNet) and [RTNet](https://github.com/OliverRensu/RTNet)
* AR: This repository heaviliy borrows from [SlowFast](https://github.com/facebookresearch/SlowFast), [TimeSformer](https://github.com/facebookresearch/TimeSformer), and [IIN](https://github.com/CompVis/iin)
