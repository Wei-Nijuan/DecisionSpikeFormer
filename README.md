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
Our Decision ConvFormer code is based on [Decision Transformer](https://github.com/kzl/decision-transformer)

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
