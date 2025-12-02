<br>
<p align="center">
<h1 align="center"><strong>SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model
</strong></h1>
  <p align="center">
      <strong><span style="color: red;">CVPR 2025</span></strong>
    <br>
   Chunlin Yu*</a>&emsp;
    <a href='https://hq-King.github.io' target='_blank'>Hanqing Wang*</a>&emsp;
   Ye Shi</a>&emsp;
   Haoyang Luo</a>&emsp;
    Sibei Yang</a>&emsp;
   Jingyi Yu</a>&emsp;
   Jingya Wang</a>&emsp;
    <br>
    ShanghaiTech University    
    <br>
    *Indicates Equal Contribution
    <br>
  </p>
</p>

  

<p align="center">
  <a href="https://seq-afford.github.io"><b>📖 Project Page</b></a> |
  <a href="https://arxiv.org/pdf/2412.01550"><b>📄 Paper Link</b></a> |
</p>

</div>

> We introduce SeqAfford, a Multi-Modal Language Model (MLLM) capable of serialized affordance inference implied in human instructions: 1) Single Affordance Reasoning; 2) Sequential Affordance Reasoning; 3) Sequential Affordance Reasoning with Multiple Objects

<div align="center">
    <img src="fig1.png" height=500>
</div>

## 📣 News
- [2/27/2025] 🎉🎉🎉SeqAfford has been accepted by CVPR 2025!!!🎉🎉🎉
- [12/2/2024] SeqAfford has been released on Arxiv now!!!

## 😲 Results
Please refer to our [homepage](https://seq-afford.github.io) for more thrilling results!


## 🛠️ Setup
- 1. Create a new `conda` environment and activate it by following command
  ```bash
  conda env create -f environment.yaml
  ```
- 2. Down [ShapeLLM](https://github.com/qizekun/ShapeLLM/blob/main/docs/MODEL_ZOO.md) model weight into your directory, and Modify the model path in the `scripts/finetune_lora.sh`， including both `--vision_tower_path` and `--pretrain_mm_mlp_adapter`
 
 - 3. Down [Uni3D](https://github.com/baaivision/Uni3D) model weight into your directory, and Modify the model path in the `./llava/model/language_model/affordancellm.py`
 
- 4. you can train your own model by running the following code
 
```bash
  sh ./scripts/finetune_lora.sh
```
# 📚 Data
visit the link to download the [Dataset](https://pan.baidu.com/s/1_koVmNMdv5BByli97eDHGA?pwd=2025)

## 🚩 Plan
- [x] Paper Released.
- [√ ] Source Code and Pretrained Weights.
- [√ ] Dataset.
<!-- --- -->

## Acknowledgement
Thanks for the wonderful works: [ShapeLLM](https://github.com/qizekun/ShapeLLM/blob/main/docs/MODEL_ZOO.md), [LISA](https://github.com/dvlab-research/LISA), This work is built upon them.

## 🎫 License

For academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 

## 🖊️ Citation
```
@article{yu2024seqafford,
        title={SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model},
        author={Yu, Chunlin and Wang, Hanqing and Shi, Ye and Luo, Haoyang and Yang, Sibei and Yu, Jingyi and Wang, Jingya},
        journal={arXiv preprint arXiv:2412.01550},
        year={2024}
      }

```
