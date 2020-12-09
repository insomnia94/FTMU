## Prerequisites

* Python 3.7.5
* Pytorch 1.3.1
* torchvision 0.4.2
* CUDA 10.1

You can download the pre-trained model  [here](https://drive.google.com/file/d/1Pz5YVwRllyS6U1gkSI8dupQuRDL-hnGe/view?usp=sharing), the training log file [here](https://drive.google.com/file/d/1Pz5YVwRllyS6U1gkSI8dupQuRDL-hnGe/view?usp=sharing), and the segmentation results [here](https://drive.google.com/file/d/1Pz5YVwRllyS6U1gkSI8dupQuRDL-hnGe/view?usp=sharing).

Please modify the dataset_root in parameter.py for the dataset path. The model will be evaluated after each eopch. The log file is saved in ./log, and the segmentation results are saved in ./best_results.

```bash
python ./train_eval.py
```

You can also see the original AC training code in AC_train.py
```bash
python ./AC_train.py
```
