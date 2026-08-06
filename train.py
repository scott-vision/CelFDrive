"""Example experiment-specific Ultralytics training invocation.

Copy this script and replace the dataset, output, device, and hyperparameter
values for a versioned experiment; importing it starts a training run.
"""

from ultralytics import YOLO

model = YOLO('Models/Base/yolov9c.pt')  
dropout = 0.0
device = [0]
# Train the model in parralel using export MKL_SERVICE_FORCE_INTEL=1
results = model.train(imgsz=640, batch=32, epochs=150, patience=50, mosaic=0.0, flipud =0.5, dropout =dropout,  data="Yaml/ScottLabelsV9_0418VAL_40xnoval.yaml", project="yolov9", name=f"20240512_ScottlabelsV9_40xnoval_{str(dropout)}_",  device=device)
