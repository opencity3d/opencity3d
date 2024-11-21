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
conda env create --file environment.yml

# Activate environment
conda activate opencity
```

## Pipeline

**Dataset Generation**

- Get the mesh by Blender using [Blosm](https://github.com/vvoovv/blosm). By following the guide you could select the place and paste the coordinate into Blender. 

- Export the mesh with .glb file and render with [Meshlab](https://www.meshlab.net/). Save the object file (.obj) together with the texture images.

- Create a ``data` folder and a folder under `data` with scene name. To obtain the RGB and depth image, you should run `generate_dataset.py`

Your dataset should have the following structure:

```
<data>
|---scane_name
|   |---scene.glb
|   |---scene.obj
|   |---texture_0.jpg
|   |---texture_1.jpg
|   |---...
```

First, optionally change the locaction of the file and output path in `generate_dataset.py`:
```
'''
Example of how to generate RGB and depth image in generate_dataset.py
'''

......
......

# Change here for your own scene generation
file = "/path/to/your/data/scene_name/scene.obj"
output_path = "/path/to/your/data/scene-output-v1/"

......
......
```

Then run the `generate_dataset.py' by:
```
cd dataset_generation
python generate_dataset.py
```
By doing this, you could get the rendered RGB-D and depth images and thus finish dataset generation:
```
<data>
|---scene_name
|   |---scene.glb
|   |---scene.obj
|   |---texture_0.jpg
|   |---texture_1.jpg
|   |---...
|---scene-output-v1
|   |---color
|   |   |---0.jpg
|   |   |---1.jpg
|   |   |---...
|   |---depth
|   |   |---0.npy
|   |   |---1.npy
|   |   |---...
|   |---intrinsic
|   |   |---intrinsic_color.txt
|   |   |---projection_matrix.txt
|   |---pose
|   |   |---0.txt
|   |   |---1.txt
|   |   |---...
```

**Piepline**

- **Step 1:** **Generate image features.**  Run the following code:

```
cd .. # Back to project root folder
cd preprocessing
python preprocess.py --dataset_path $path-of-scene-output-v1 --model siglip --mode highlight
```
To run the baseline (Openscene/LangSplat + CLIP) you should run `preprocess_level0.py` instead of `preprocess.py`.

**ATTENTION!** This step takes a lot time, after that you may find the generated feature under the folder `/scene-output-v1/` with name `language_features_highlight` (Without highlight for running baseline)

- **Step 2:** Projecting features to scene and generate point cloud.

    - Adjust the path to the scene mesh and language features generated:
    ```
    '''
    Example of how to generate RGB and depth image in convert_to_point_cloud.py
    '''

    ......
    ......

    if __name__ == "__main__":
        if True:
            base_path = "/path/to/your/data/scene-output-v1/"
            obj_path = "/path/to/your/data/scene_name/scene.obj"
            full_embeddings_mode = False # True if you are doing baseline
            
        convert_to_pcd(obj_path = obj_path, #"scene_example_downsampled.ply",
                        images_path= base_path + "color",
                        depth_path = base_path + "depth",
                        feat_path = base_path + "language_features",
                        mask_path = base_path + "language_features",
                        full_embedding_path = base_path + "full_image_embeddings",
                        poses_path = base_path + "pose",
                        intrinsics_path = base_path + "intrinsic/projection_matrix.txt",
                        output_path = "semantic_point_cloud.ply",
                        full_embeddings_mode = full_embeddings_mode)
    ```

    - Running the code:
    ```
    cd ..
    python convert_to_point_cloud.py
    ```
    You may find the generated features and point cloud file (`point_features_highlight.npy` and `generated_point_cloud.ply`) under the `/eval` folder.

## Example

We prepare the Rotterdam scene's process result, which contains the generated point cloud and extracted highlighted features. You can download via the [link](https://drive.google.com/drive/folders/1kVBiNlEXEGp3iBkMb7EQ5NT-CFLq1vVx) and play with the `visualize_pcd_features.ipynb` under sandbox folder.

### How to play with:

- Create a folder under `/data/` with the name `/embedded_point_clouds/`. Then create a `scene_name` sub-folder under `/embedded_point_clouds/`.

- Put the generated feature and point cloud files into it. Then change the following configuration in the notebook:
```
tag = "scene_name" # name of the sub-folder
model_type = "siglip"
crop_type = "highlight" #"highlight" #"full"
```

- Run the following uncommented notebook to download the siglip model and tokens. 
    - E.g. Running this block for visualizing the heat scene of query result.
```
queries = ["tree"] # Set the query here
query_embed = encode_text(queries, model, tokenizer)
sim = features @ query_embed 
sim = sim.max(axis=1)
# sim = np.exp(sim)
    # sim = np.exp(sim) / (np.exp(sim) + np.exp(max_canonical_sim))
print(sim.shape)

for i, query in enumerate(queries):
    visualize(pcd, sim[:,i], query)
```

## TODO list:
- [ ] Update Readme
- [ ] release the arhxiv camera-ready version
- [ ] release the code of the embedding training
- [ ] release the preprocessed dataset and the pretrained embeddings
- [ ] release the code of the visulization cookbook
- [ ] release the code of experienment tasks