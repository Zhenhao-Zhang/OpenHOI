# [NeurIPS 2025 Oral] OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Model

This is the offical code repo for **NeurIPS 2025 Oral** paper **OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Model**

[[paper]](https://arxiv.org/abs/2505.18947) [[project page]](http://openhoi.github.io/)

<div align="center">
    <img src="pipeline.png" height=500>
</div>

# Disclaimers
- Code Quality Level: Tired grad student, lots of hard code in my repo
- Training Enviroment: HOIAffordanceMLLM: A800 80G GPUs(use 73G). Affordance-Driven HOI Diffusion: 4090 24G GPUs
- Questions: please drop me an email, it is the fastest way to get feedback
- For Enviroment Set Up: I set the enviroment in my gpus by this way, may have more easy ways. But I believe you can set up the enviroment by my steps

# Plan
- [√ ] Paper Released.
- [√ ] Code.
- [√ ] Pretrained Weights.
- [√ ] Dataset.
- [√ ] Quick Start
- [√ ] Weights of HOIAffordanceMLLM
- [√ ] Weights of Affordance-Driven HOI Diffusion

Any Question, feel free to contact zhangzhh2024@shanghaitech.edu.cn


# Set Up Enviroment for HOIAffordanceMLLM
- 1. Create Python Enviroment

    - 1.1. Create Conda Enviroment

      ```
      conda create -n HOIAffordanceMLLM python=3.10
      ```

    - 1.2. Get Pytorch-GPU

      ```
      pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
      ```

    - 1.3 Install KNN-Cuda

      ```
      pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
      ```

    - 1.4 Install Pointnet++

      ```
      pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
      ```

      **If fail:**
      ```
      cd ~

      git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git

      cd Pointnet2_PyTorch

      pip install -r requirements.txt

      pip install -e .
      ```
    - 1.5 Install torch-scatter
      ```
      pip install torch-scatter==2.0.9 --no-build-isolation
      ```
    - 1.6 Install llava

      ```
      cd /yourpath/HOIAffordanceMLLM
      pip install -e .
      ```


    - 1.7 Get Other Python Packages

      ```
      pip install -r requirements.txt
      ```

- 2. Down [ShapeLLM](https://github.com/qizekun/ShapeLLM/blob/main/docs/MODEL_ZOO.md) model weight and json into your directory, and Modify the model path in the `scripts/finetune_lora.sh`， including both `--vision_tower_path` and `--pretrain_mm_mlp_adapter`, and `LLM_VERSION`

      **Tip**: Replace `/root/tmp` with your path
      ```
      pip install -U huggingface_hub

      export HF_ENDPOINT=https://hf-mirror.com

      huggingface-cli download --resume-download qizekun/ReConV2 --local-dir /root/tmp --include "zeroshot/large/best_lvis.pth"
      
      mkdir ShapeLLM_7B_gapartnet_v1.0

      huggingface-cli download --resume-download qizekun/ShapeLLM_7B_gapartnet_v1.0 --local-dir /root/tmp/ShapeLLM_7B_gapartnet_v1.0 

      mkdir shapellm

      huggingface-cli download --repo-type dataset --resume-download qizekun/ShapeLLM --local-dir /root/tmp/shapellm --include "gapartnet_sft_27k_openai.json"
      
      huggingface-cli download --repo-type dataset --resume-download qizekun/ShapeLLM --local-dir /root/tmp/shapellm --include "gapartnet_pcs.zip"
      
      bash scripts/extract_mm_projector.py
      You can also download mm_projection.bin there: https://pan.baidu.com/s/1TFjp8n9JhonxUdaUms2vcw?pwd=ia8m 
    
      ```
- 3. Down [Uni3D](https://github.com/baaivision/Uni3D) model weight into your directory, and Modify the model path in the `./llava/model/language_model/affordancellm.py`

      ```
      mkdir uni3d
      huggingface-cli download --repo-type dataset --resume-download BAAI/Uni3D --local-dir /root/tmp/uni3d --include "modelzoo/uni3d-b/model.pt"
      ```


 
<!-- - 4. If you want to get multi-object affordance or scene-level manipulation for long-horizon hoi, try the multi-object-affordance/scene_planning.py -->

# Set Up Enviroment for Affordance-Driven HOI Diffusion

- 1. Create Python Enviroments
      ```
      conda create -n openhoi python=3.8 -y
      conda activate openhoi
      ```
- 2. Get pyyaml
      ```
      pip install pyyaml==6.0.1
      ```
- 3. Install pytorch3d 0.7.2
      ```
      conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
      conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
      conda install -c bottler nvidiacub -y
      pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu116_pyt1130/download.html
      ```
- 4. Get other requirements
      ```
      pip install -r requirements.txt
      ```
- 5. Get spacy
      ```
      python -m spacy download en_core_web_sm
      ```
- 6. Get CLIP
      ```
      pip install git+https://github.com/openai/CLIP.git
      ```
- 7. Get numpy
      ```
      pip install numpy==1.23.5
      ```
- 8. Download Pretrain Weights on [Download](https://drive.google.com/drive/folders/1bfYF94-dVy-mA0n4cIRb_wI4ohPC6KK5?usp=sharing)
# Data Prepare
- 1. Download the affordance dataset and hoi dataset:

Afforodance Dataset: [Download](https://pan.baidu.com/s/1Wsz6YK-IJ6yQ80dsYPdqrA?pwd=6swx)                                        
 

HOI Dataset: [GRAB](https://grab.is.tue.mpg.de/index.html) [GRAB Text](https://drive.google.com/drive/folders/1vQXrplvS9fukMqHBH7JOne5DoaqTCL5w?usp=sharing) [ARCTIC](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md#download-full-arctic) [ARCTIC Text](https://drive.google.com/drive/folders/1vQXrplvS9fukMqHBH7JOne5DoaqTCL5w?usp=sharing)

- 2. You should process your pointcloud to 2048*3 for both training and inference time:
```
python DataProcess/point_cloud_process.py
python Affordance-DrivenHOIDiffusion/preprocessing.py
```
- 3. Process the instrutions to open-vocabulary instructions:
```
python DataProcess/high_level_instructions.py
```

# Weights
Weights for HOIAffordanceMLLM: [https://pan.baidu.com/s/13yP3ihztAcBF35JYMvHD8w?pwd=q6z3](Download)

Weights for Affordance-Driven HOI Diffusion: In output folder texthom_best.pt

# Quick Start
- 1. Coarse Fine-tuning the HOIAffordanceMLLM with Affordance Dataset

      ```
      bash HOIAffordanceMLLM/scripts/finetune_lora.sh
      ```

- 2. Fine-grained Aligenment for Hand-Object Contact:(This step will transfer object-centric affordance to hand-centric affordance(contact map, with only 0/1))
      ```
      bash Affordance-DrivenHOIDiffusion/scripts/train/train_contact_estimator.sh
      ```
- 3. Train Affordance-Driven HOI Diffusion
      ```
      bash Affordance-DrivenHOIDiffusion/scripts/train/train_texthom.sh
      ```

- 4. OpenHOI Quick Inference
      ```
      python Affordance-DrivenHOIDiffusion/start/inference.py
      ```

# Acknowledgement
Thanks for the excellent work [ShapeLLM](https://github.com/qizekun/ShapeLLM/),[Text2HOI](https://github.com/JunukCha/Text2HOI),[DSG](https://github.com/LingxiaoYang2023/DSG2024),[SeqAfford](https://github.com/hq-King/SeqAfford),[GazeHOI](https://github.com/takiee/GazeHOI-toolkit)


# Citation

If you find our work useful in your research, please consider citing

```
@misc{zhang2025openhoiopenworldhandobjectinteraction,
      title={OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Model}, 
      author={Zhenhao Zhang and Ye Shi and Lingxiao Yang and Suting Ni and Qi Ye and Jingya Wang},
      year={2025},
      eprint={2505.18947},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.18947}, 
}
```