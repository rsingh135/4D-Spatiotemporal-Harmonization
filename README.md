# Segment-Anything-in-4D
This project is obtained by organizing and supplementing the unorganized code based on the paper "Segment Any 4D Gaussians".

## Environmental Setups

```ba
git submodule update --init --recursive
conda create -n sa4d python=3.9
conda activate sa4d
conda install -c conda-forge colmap

pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/diff-gaussian-rasterization_contrastive_f
pip install -e submodules/simple-knn

pip install "git+https://github.com/facebookresearch/pytorch3d.git"

cd Tracking-Anything-with-DEVA
pip install -r requirements.txt
git clone ssh://git@github.com/IDEA-Research/Grounded-Segment-Anything
cd Grounded-Segment-Anything
pip install -e .
cd ../..

In our environment, we use pytorch=2.0.1+cu118.
```

## Training
### Data Preprocess

For training hypernerf scenes such as `virg/broom`: Pregenerated point clouds by COLMAP are provided [here](https://drive.google.com/file/d/1fUHiSgimVjVQZ2OOzTFtz02E9EqCoWr5/view). Just download them and put them in to correspond folder, and you can skip the former two steps. Also, you can run the commands directly.

```python
# First, computing dense point clouds by COLMAP
bash colmap.sh data/hypernerf/broom2 hypernerf
# Second, downsample the point clouds generated in the first step. 
python scripts/downsample_point.py data/hypernerf/virg/broom2/colmap/dense/workspace/fused.ply data/hypernerf/virg/broom2/points3D_downsample2.ply                                                      
```

For training dynerf scenes：

```bash
# First, extract the frames of each video.
python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
# Second, generate point clouds from input data.
bash colmap.sh data/dynerf/cut_roasted_beef llff
# Third, downsample the point clouds generated in the second step.
python scripts/downsample_point.py data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply data/dynerf/cut_roasted_beef/points3D_downsample2.ply
```

### Label

```bash
bash prepare_pseudo_label.sh ./data/hypernerf/broom2/ 0
bash prepare_pseudo_label.sh ./data/dynerf/cut_roasted_beef 0
```

### Train

```bash
# split-cookie
python train_4dgs.py -s ./data/hypernerf/split-cookie/ --port 6017 --expname "hypernerf/split-cookie" --configs arguments/hypernerf/default.py
python render_4dgs.py --model_path "output/hypernerf/split-cookie/" --skip_train --skip_test --configs arguments/hypernerf/default.py
python train_ie.py -s ./data/hypernerf/split-cookie/ -m ./output/hypernerf/split-cookie/ --configs arguments/hypernerf/default.py
python render_ie.py --model_path "output/hypernerf/split-cookie/"  --skip_train --skip_test --configs arguments/hypernerf/default.py
```

See `command.sh` more often, where there are more training commands for the hypernerf and dynerf datasets.

## Demo

`delete.ipynb`: Specify to delete the object. The ID of the object has been commented inside. If you want to obtain it, you can uncomment.

`composite.ipynb`: How to combine multiple scenes into the same space.

`demo_dynerf.ipynb`，`demo_hypernerf.ipynb`:Interactive segmentation is carried out using the SAM model, the segmentation effect is improved through loss optimization, and the segmentation results are verified from different perspectives.

`demo.ipynb`: Extract the corresponding object ID and display it.





