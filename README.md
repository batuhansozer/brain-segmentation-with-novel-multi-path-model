# Brain Segmentation with Novel Multi Path Model
[![DOI](https://zenodo.org/badge/379040837.svg)](https://zenodo.org/badge/latestdoi/379040837)
## Overview
This repository contains source code for multi path brain segmentation model that uses T1-Weighted, T2-Weighted and FLAIR MRI images as input. The architecture is using U-Net architecture with residual extended skip blocks as baseline model. Model is trained and tested with Gazi Brains 2020 dataset.

## Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python create_dataset.py --input_width 224 --input_height 224 --dataset_path Data/gazi_brains --dataset_save_path Data/npy_dataset

python split_npy_dataset.py --dataset_path Data/npy_dataset --split_dataset_save_path Data/split_npy_dataset --batch_size 32

# To train baseline model
python baseline_model_train.py --use_gpu True --input_height 224 --input_width 224 --split_npy_dataset_path Data/split_npy_dataset --epochs 100

# To train multi channel model
python multi_channel_model_train.py --use_gpu True --input_height 224 --input_width 224 --split_npy_dataset_path Data/split_npy_dataset --epochs 100

# To train proposed multi path model
python multi_path_model_train.py --use_gpu True --input_height 224 --input_width 224 --split_npy_dataset_path Data/split_npy_dataset --epochs 100

python predict.py --use_gpu True --dataset_path Data/npy_dataset --baseline_model Model/single_channel.h5 --multi_path_model Model/multi_encoder.h5 --multi_channel_model Model/multi_channel.h5

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[GNU LGPLv3](https://choosealicense.com/licenses/lgpl-3.0/)
