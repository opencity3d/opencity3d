<p align="center">
  <h1 align="center">OpenCity3D🏙️: Open-Vocabulary 3D Instance Segmentation</h1>
<!-- # OpenCity3D: What do Vision-Language Models know about Urban Environments? (WACV 2025) -->
    <p align="center">
        <a>Valentin Bieri</a><sup>1</sup>, &nbsp;&nbsp;&nbsp; 
        <a>Marco Zamboni</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
        <a>Nicolas S Blumer</a><sup>1,2</sup>&nbsp;&nbsp;&nbsp; 
        <a href="https://jerryisqx.github.io/">Qingxuan Chen</a><sup>1,2</sup>
        <br>
        <a href="https://francisengelmann.github.io/">Francis Engelmann</a><sup>1,3</sup>
        </br>
        <br>
        <sup>1</sup>ETH Zürich&nbsp;&nbsp;&nbsp;&nbsp;
        <sup>2</sup>University of Zurich&nbsp;&nbsp;&nbsp;&nbsp;
        <sup>3</sup>Stanford University&nbsp;&nbsp;&nbsp;&nbsp;
        </br>
    </p>
    <h2 align="center">WACV 2025</h2>
    <p align="center">
        <a href=""><img alt="arXiv" src="https://img.shields.io/badge/arXiv-badge"> </a>
        <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    </p>
    <h3 align="center"><a href="">Paper</a> | <a href="https://opencity3d.github.io">Project Page</a>
    </h3>
</p>



<!-- <a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a> -->

![teaser](https://opencity3d.github.io/static/images/teaser.jpg)

<p align="center">
<strong>OpenCity3D</strong> is a zero-shot approach for open-vocabulary 3D urban scene understanding.
</p>

### BibTex
```
@inproceedings{opencity3d2025,
    title = {OpenCity3D: 3D Urban Scene Understanding with Vision-Language Models},
    author = {Bieri, Valentin and Zamboni, Marco and Blumer, Nicolas S. and Chen, Qingxuan and Engelmann, Francis},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year = {2025},
    organization = {IEEE}
}
```

---

## Setup Environment

Please clone this repository first with running:
```
git clone https://github.com/opencity3d/opencity3d.git
```

Preparing Conda environment:
```
# Create environment
conda create -n opencity python
# Install dependencies
pip install -r requirements.txt

# Activate environment
conda activate opencity
```

## Pipeline

**Dataset Generation**

**Training Embedding**

**Projecting**

## Dataset Structure



## TODO list:
- [ ] Update Readme
- [ ] release the arhxiv camera-ready version
- [ ] release the code of the embedding training
- [ ] release the preprocessed dataset and the pretrained embeddings
- [ ] release the code of the visulization cookbook
- [ ] release the code of experienment tasks