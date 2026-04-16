import torch
path = 'output/hypernerf/split-cookie/point_cloud/iteration_14000/classifier.pt'
checkpoint = torch.load(path, map_location='cpu')
print("\n--- Classifier Structure ---")
for k, v in checkpoint.items():
    if hasattr(v, 'shape'):
        print(f"Key: {k:15} | Shape: {list(v.shape)}")
