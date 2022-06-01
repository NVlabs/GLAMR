# Generating Dynamic Human3.6M

## Step 1: Download Human3.6M from the official [website](http://vision.imar.ro/human3.6m)
## Step 2: Preprocess Human3.6M into COCO format
We use the same COCO format for Human3.6M as [PoseNet](https://github.com/mks0601/3DMPPE_POSENET_RELEASE) and [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE).  
To generate the format, follow the [instructions](https://github.com/mks0601/3DMPPE_POSENET_RELEASE/tree/master/tool/Human36M) in the PoseNet repo:
1. Run the matlab preprocessing [script](https://github.com/mks0601/3DMPPE_POSENET_RELEASE/blob/master/tool/Human36M/preprocess_h36m.m) using the official [Human3.6M SDK](http://vision.imar.ro/human3.6m).
2. Run [h36m2coco.py](https://github.com/mks0601/3DMPPE_POSENET_RELEASE/blob/master/tool/Human36M/h36m2coco.py):
    ```
    python h36m2coco.py
    ```
3. Download the [SMPL parameters](https://drive.google.com/file/d/1cGxpedigRYMYmd_Qckc2l801aEkSLc9f/view?usp=sharing) obtained using gradient-based optimization. Unzip the file into `datasets/smpl_fit`.

The resulting COCO-format Human3.6m dataset will have the following structure: 
```
${GLAMR_ROOT}
|-- datasets
|   |-- H36M
|   |   |-- images
|   |   |-- annotations
|   |   |-- smpl_fit
```
## Step 3: Generate Dynamic Human3.6M
Take the following steps in the root folder of this repo:
1. Further process the COCO-format human3.6M dataset:
    ```
    python preprocess/preprocess_h36m.py
    ```
1. Generate Dynamic Human3.6M with occlusions:
    ```
    python preprocess/preprocess_h36m_occluded.py
    ```
The resulting Dynamic Human3.6M dataset is stored in `datasets/H36M/occluded_v2`.
