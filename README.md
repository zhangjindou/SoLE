# SoLE

Enhanced Knowledge Graph Embedding by Jointly Learning Soft Rules and Facts.

Paper is published in MDPI Algorithms: [download here](https://www.mdpi.com/1999-4893/12/12/265).

### Statement

We implement the embedding model of our algorithm SoLE based on the open resource code [tensorflow-efe](https://github.com/billy-inn/tensorflow-efe), where it trains the models by both training and validation sets while SoLE only uses training sets. 

Thank the author of tensorflow-efe very much for sharing it.

### Requirements

- Python 3
- Tensorflow >= 1.2
- Hyperopt
- JBoss Drools 6.2.0

### Datasets

Datasets used in our paper are stored in the `Datasets` directory, including FB15K, DB100K and FB15K-sparse.

### Grounding Generation Stage

#### Rule Mining

`java -jar amie_plus.jar -maxad 3 -minpca 0.8 -minhc 0.8 ./Datasets/fb15k/train.txt `

Rules extracted by [AMIE+](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/amie/) are stored in the file `rule_[confidence].txt` which can be found in the corresponding dataset directory.

#### Forward Chaining Reasoning

The project `GenGroundings` functions this module. After executing [GroundAllRulesByRE.java](https://github.com/zhangjindou/SoLE/blob/master/GenGroundings/src/GroundAllRulesByRE.java), it will performs the reasoning and generates the groundings in the file `groundings_[confidence].txt` or `groundings_oneTime_[confidence].txt`.

### Embedding Learning Stage

#### Configurations

The configurations of datasets can be set in [config.py](https://github.com/zhangjindou/SoLE/blob/master/EmbLearning/config.py)

#### Hyperparameters

Add hyperparameters dict and its identifier in [model_param_space.py](https://github.com/zhangjindou/SoLE/blob/master/EmbLearning/model_param_space.py).

### Evaluation

`CUDA_VISIBLE_DEVICES=[gpu] python train.py -m [model_name] -d [data_name]`

Train on the given hyperparameter setting and give the result for the test set.


### Cite

If you find our work useful, please cite:
```
@article{zhang2019enhanced,
title={Enhanced Knowledge Graph Embedding by Jointly Learning Soft Rules and Facts},
author={Zhang, Jindou and Li, Jing},
journal={Algorithms},
volume={12},
number={12},
pages={265},
year={2019}}
```


### License

MIT