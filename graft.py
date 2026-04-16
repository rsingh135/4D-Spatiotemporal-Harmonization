import torch
import os
from os import path
from utils.system_utils import mkdir_p
from scene import GaussianModel

# Simple class to mimic the 'args' object the model expects
class DummyArgs:
    def __init__(self):
        self.sh_degree = 3
        self.source_path = ""
        self.model_path = ""
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"

def merge():
    # 1. Setup Paths
    stage_path = "output/hypernerf/split-cookie/point_cloud/iteration_14000/point_cloud.ply"
    asset_path = "output/hypernerf/torchocolate/point_cloud/iteration_14000/point_cloud.ply"
    output_dir = "output/composite/grafted_scene/point_cloud/iteration_14000"
    mkdir_p(output_dir)

    # 2. Initialize Models
    args = DummyArgs()
    stage_gaussians = GaussianModel(sh_degree=3, mode='scene', args=args) 
    stage_gaussians.load_ply(stage_path)
    
    asset_gaussians = GaussianModel(sh_degree=3, mode='scene', args=args)
    asset_gaussians.load_ply(asset_path)

    # 3. Load Identity Encodings (Features)
    stage_data = torch.load("output/hypernerf/split-cookie/point_cloud/iteration_14000/features.pth")
    asset_data = torch.load("output/hypernerf/torchocolate/point_cloud/iteration_14000/features.pth")
    
    stage_ids = stage_data['obj_ids']
    asset_ids = asset_data['obj_ids']

    # 4. Filter the points
    # Keep everything in stage EXCEPT the Cookie (ID 5)
    stage_mask = (stage_ids != 5).squeeze()
    # Keep ONLY the Chocolate (ID 3) from the asset
    asset_mask = (asset_ids == 3).squeeze()

    # 5. Combine Attributes
    # You can tweak this offset later if the chocolate is floating/buried
    offset = torch.tensor([0.0, 0.0, 0.0]).to("cuda")
    
    new_xyz = torch.cat([stage_gaussians._xyz[stage_mask], asset_gaussians._xyz[asset_mask] + offset], dim=0)
    new_features_dc = torch.cat([stage_gaussians._features_dc[stage_mask], asset_gaussians._features_dc[asset_mask]], dim=0)
    new_features_rest = torch.cat([stage_gaussians._features_rest[stage_mask], asset_gaussians._features_rest[asset_mask]], dim=0)
    new_opacity = torch.cat([stage_gaussians._opacity[stage_mask], asset_gaussians._opacity[asset_mask]], dim=0)
    new_scaling = torch.cat([stage_gaussians._scaling[stage_mask], asset_gaussians._scaling[asset_mask]], dim=0)
    new_rotation = torch.cat([stage_gaussians._rotation[stage_mask], asset_gaussians._rotation[asset_mask]], dim=0)

    # 6. Set the new values into the stage model and save
    stage_gaussians._xyz = torch.nn.Parameter(new_xyz)
    stage_gaussians._features_dc = torch.nn.Parameter(new_features_dc)
    stage_gaussians._features_rest = torch.nn.Parameter(new_features_rest)
    stage_gaussians._opacity = torch.nn.Parameter(new_opacity)
    stage_gaussians._scaling = torch.nn.Parameter(new_scaling)
    stage_gaussians._rotation = torch.nn.Parameter(new_rotation)
    
    stage_gaussians.save_ply(path.join(output_dir, "point_cloud.ply"))
    
    # Save a dummy features.pth so the renderer doesn't crash later
    new_features = torch.cat([stage_data['features'][stage_mask], asset_data['features'][asset_mask]], dim=0)
    new_ids = torch.cat([stage_ids[stage_mask], asset_ids[asset_mask]], dim=0)
    torch.save({'features': new_features, 'obj_ids': new_ids}, path.join(output_dir.replace("point_cloud/iteration_14000", ""), "features.pth"))

    print(f"✅ Composition complete! Saved to {output_dir}")

if __name__ == "__main__":
    merge()
