# Preparing ReCon1M Dataset

>[ReCon1M: A Large-Scale Benchmark Dataset for Relation Comprehension in Remote Sensing Imagery](https://arxiv.org/pdf/2406.06028)

>[ReCon1M project page](https://recon1m-dataset.github.io/)

>[ReCon1M IEEE TGRS](https://ieeexplore.ieee.org/document/11082556)


## Download ReCon1M dataset

The STAR dataset can be downloaded from  [official baidu netdist](https://pan.baidu.com/s/1B-DLJDN3YertZyp8E7UtuQ) Access code: mcq6 or [modelscope(魔塔)](https://www.modelscope.cn/datasets/wokaikaixinxin/ReCon1M/files).

**How to use modelscope(魔塔) to download ReCon1M**

1) Install `modelscope`

```shell
pip install modelscope
```

2) Download ReCon1M

```shell
modelscope download --dataset 'wokaikaixinxin/ReCon1M' --local_dir 'your_local_path'
```

The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── ReCon1M
│   │   ├── dataset_split.json
│   │   ├── images.zip
│   │   ├── labelTxt.zip
│   │   ├── rearrange_dataset.py
│   │   ├── Recon1m_info.json
│   │   ├── relTxt.zip
```


## Classes of ReCon1M

The 60 classes.

```
'classes':
(
 'van', 'small car', 'building', 'road', 'airplane', 'block', 'parking lot', 'motorboat', 'dump truck', 'cargo truck', 'dry cargo ship', 'runway', 'container', 'water', 'intersection', 'fishing boat', 'other vehicle', 'storage tank', 'airport', 'other ship', 'harbor', 'engineering ship', 'tennis court', 'pool', 'solar panel', 'liquid cargo ship', 'crane', 'bus', 'passenger ship', 'warship', 'storage tank group', 'excavator', 'bridge', 'tugboat', 'basketball court', 'trailer', 'train carriage', 'football field', 'cargo', 'baseball field', 'exhaust fan', 'truck tractor', 'factory', 'roundabout', 'construction site', 'chimney', 'stadium', 'smoke', 'railway', 'boarding bridge', 'farmland', 'helipad', 'tractor', 'greenbelt', 'control tower', 'dam', 'typhoon spiral', 'typhoon eye', 'locomotive', 'gas-station'
)
```



## Description

ReCon1M is the first million-level relation annotation dataset specifically designed for Scene Graph Generation (SGG) in remote sensing imagery. our dataset is built upon FAIR1M and comprises 22,262 images. It includes annotations for 873,761 object bounding boxes across 60 categories, and 1,052,244 relation triplets across 59 categories based on these bounding boxes.

[Paper link](https://arxiv.org/pdf/2406.06028)

<div align=center>
<img src="https://recon1m-dataset.github.io/images/example.png" />
</div>


```bibtex
@ARTICLE{11082556,
  author={Yan, Qiwei and Deng, Chubo and Liu, Chenglong and Hou, Zhongyan and Liu, Xiaorui and Jiang, Yi and Lu, Wanxuan and Yao, Fanglong and Liu, Xiaoyu and Hao, Lingxiang and Yu, Hongfeng and Sun, Xian},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={ReCon1M: A Large-Scale Benchmark Dataset for Relation Comprehension in Remote Sensing Imagery}, 
  year={2025},
  volume={63},
  number={},
  pages={1-22},
  keywords={Remote sensing;Visualization;Annotations;Cognition;Semantics;Benchmark testing;Training;Artificial intelligence;Data mining;Computational modeling;Benchmark dataset;relation comprehension;remote sensing;scene graph generation (SGG)},
  doi={10.1109/TGRS.2025.3589986}}
```