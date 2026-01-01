# Preparing STAR Dataset

>[STAR: A First-Ever Dataset and A Large-Scale Benchmark for Scene Graph Generation in Large-Size Satellite Imagery](https://arxiv.org/abs/2406.09410)

>[STAR project page](https://linlin-dev.github.io/project/STAR)

>[STAR-MMRotate](https://github.com/VisionXLab/STAR-MMRotate)

>[SGG-ToolKit](https://github.com/Zhuzi24/SGG-ToolKit)


## Download STAR dataset

The STAR dataset can be downloaded from [official hugging face](https://huggingface.co/datasets/Zhuzi24/STAR) or [official baidu netdist](https://pan.baidu.com/s/143LVm6zt-ryGEngltALZtw?pwd=STAR) or [modelscope(魔塔)](https://modelscope.cn/datasets/wokaikaixinxin/STAR/files).

**How to use modelscope(魔塔) to download STAR**

1) Install `modelscope`

```shell
pip install modelscope
```

2) Download STAR

```shell
modelscope download --dataset 'wokaikaixinxin/STAR' --local_dir 'your_local_path'
```

The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── STAR
│   │   ├── train
│   │   │   │   ├── img (771 png, 64GB)
│   │   │   │   ├── object-TXT (771 txt, ~7MB)
│   │   ├── val (The original paper does not use a validation set, so you don't need to download it.)
│   │   │   │   ├── img (238 png, 32GB)
│   │   │   │   ├── object-TXT (238 txt, ~2.5MB)
│   │   ├── test
│   │   │   │   ├── img264 (264 png, 23.1GB)
│   │   │   │   ├── object-TXT (264 txt, ~2.56MB)
```

## split STAR dataset

train set

```shell
python tools/data/star/split/img_split.py --base-json tools/data/star/split/split_configs/ss_train.json
```

test set

```shell
python tools/data/star/split/img_split.py --base-json tools/data/star/split/split_configs/ss_test.json
```

Please update the `img_dirs` and `ann_dirs` in json.

The new data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── split_ss_star
│   │   ├── train
│   │   │   ├── images (70383 png, only 19756 images have gt)
│   │   │   ├── annfiles (70383 txt, only 19756 images have gt)
│   │   ├── test
│   │   │   ├── images (23702 png)
│   │   │   ├── annfiles (23702 txt)
```

Please change `data_root` in `configs/_base_/datasets/star.py` to `data/split_ss_star/`.

## Classes of STAR

The 48 classes.

```
'classes':
(
    'ship', 'boat', 'crane', 'goods_yard', 'tank', 'storehouse', 'breakwater', 'dock',
    'airplane', 'boarding_bridge', 'runway', 'taxiway', 'terminal', 'apron', 'gas_station',
    'truck', 'car', 'truck_parking', 'car_parking', 'bridge', 'cooling_tower', 'chimney',
    'vapor', 'smoke', 'genset', 'coal_yard', 'lattice_tower', 'substation', 'wind_mill',
    'cement_concrete_pavement', 'toll_gate', 'flood_dam', 'gravity_dam', 'ship_lock',
    'ground_track_field', 'basketball_court', 'engineering_vehicle', 'foundation_pit',
    'intersection', 'soccer_ball_field', 'tennis_court', 'tower_crane', 'unfinished_building',
    'arch_dam', 'roundabout', 'baseball_diamond', 'stadium', 'containment_vessel'
)
```

## Description

Scene graph generation (SGG) in satellite imagery (SAI) benefits promoting understanding of geospatial scenarios from perception to cognition. In SAI, objects exhibit great variations in scales and aspect ratios, and there exist rich relationships between objects (even between spatially disjoint objects), which makes it attractive to holistically conduct SGG in large-size very-high-resolution (VHR) SAI. However, there lack such SGG datasets. Due to the complexity of large-size SAI, mining triplets <subject, relationship, object> heavily relies on long-range contextual reasoning. Consequently, SGG models designed for small-size natural imagery are not directly applicable to large-size SAI. This paper constructs a large-scale dataset for SGG in large-size VHR SAI with image sizes ranging from 512 x 768 to 27,860 x 31,096 pixels, named STAR (Scene graph generaTion in lArge-size satellite imageRy), encompassing over 210K objects and over 400K triplets. To realize SGG in large-size SAI, we propose a context-aware cascade cognition (CAC) framework to understand SAI regarding object detection (OBD), pair pruning and relationship prediction for SGG. We also release a SAI-oriented SGG toolkit with about 30 OBD and 10 SGG methods which need further adaptation by our devised modules on our challenging STAR dataset.

[Paper link](https://arxiv.org/abs/2406.09410)

<div align=center>
<img src="https://github.com/VisionXLab/STAR-MMRotate/raw/main/demo/star.jpg" />
</div>


```bibtex
@article{li2025star,
  title={STAR: A first-ever dataset and a large-scale benchmark for scene graph generation in large-size satellite imagery},
  author={Li, Yansheng and Wang, Linlin and Wang, Tingzhu and Yang, Xue and Luo, Junwei and Wang, Qi and Deng, Youming and Wang, Wenbin and Sun, Xian and Li, Haifeng and others},
  journal={IEEE Trans. Pattern Anal. Mach. Intell},
  volume={47},
  number={3},
  pages={1832--1849},
  year={2025}
}
```