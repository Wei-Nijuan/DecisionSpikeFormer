## Decision SpikeFormer: Spike-Driven Transformer for Decision Making
This repository contains the official PyTorch implementation of Decision SpikeFormer: Spike-Driven Transformer for Decision Making. The paper is accepted at CVPR 2025.

## Instructions
### Installation and Dataset
Please refer to the [Decision Transformer](https://github.com/kzl/decision-transformer) for installation instructions.

### Running 
To train the Decision SpikeFormer model on the D4RL dataset, run the following command:
```bash
./gym.sh
```
### Code 
The Decision SpikeFormer code is in the `gym/models/decision_spikeformer_pssa.py`.


## Ackownledgements
Our Decision SpikeFormer code is based on [Decision Transformer](https://github.com/kzl/decision-transformer)

## Limitations

[//]: # (在我们的架构中，模型的输入需要经过一个全连接层进行预处理，输出需要经过一个全连接层进行后处理，因此导致我们的模型并不是一个全脉冲的模型。)
In our architecture, the input needs to be processed by a fully connected layer before being fed into the model 
(Embedding Layer, ANN), 
and the output needs to be processed by another fully connected layer after the model(Prediction Head, ANN).



## Citation
```
@InProceedings{huang2025decisionspike,
  author    = {Wei Huang, Qinying Gu, Nanyang Ye},
  title     = {Decision SpikeFormer: Spike-Driven Transformer for Decision Making},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  month     = {June}
}
```
