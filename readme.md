### 模式识别第三次作业

姓名：童书未		学号：21307253

------

##### 实验环境：

Linux jupyter-21307253 5.4.0-110-generic

18.04.1-Ubuntu

##### 文件结构：

```shell
.
├── 21307253-童书未-实验报告.pdf
├── data_cifar
│   ├── cifar.py
│   └── randaugment.py
├── models
│   └── WideResNet.py
├── readme.md
├── results
│   ├── test1
│   │   ├── checkpoint.pth.tar
│   │   ├── events.out.tfevents.1718614310.jupyter-21307253.57235.0
│   │   ├── events.out.tfevents.1718628341.jupyter-21307253.50665.0
│   │   └── model_best.pth.tar
│   ├── test2
│   │   ├── checkpoint.pth.tar
│   │   ├── events.out.tfevents.1718680015.jupyter-21307253.45568.0
│   │   └── model_best.pth.tar
│   └── test3
│       ├── checkpoint.pth.tar
│       ├── events.out.tfevents.1718680025.jupyter-21307253.45574.0
│       └── model_best.pth.tar
├── model_train.py
└── utils
    ├── __init__.py
    └── misc.py
```

在result文件夹中存放训练的模型：test1、test2、test3分别对应4000张、250张、40张标注数据

models文件夹：存放WideResNet-28-2训练模型

在有CUDA的环境下，运行如下指令，程序就能自行下载数据集并进行运行

```py
#4000张标注数据
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/test1
#250张标注数据
python train.py --dataset cifar10 --num-labeled 250 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/test2
#40张标注数据
python train.py --dataset cifar10 --num-labeled 40 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/test3
```

