"""DFM Studio: train a DeepFaceLive-compatible .dfm face model in PyTorch.

DeepFaceLab (TensorFlow) cannot run on RTX 50 (Blackwell) under Windows, so the
classic DFL "DF" autoencoder is reimplemented here: one shared encoder +
bottleneck, one decoder per identity. Trained to reconstruct both faces, the
"source" decoder fed with your face yields the source identity with your
expression and pose. The export has DeepFaceLive's DFM interface, so the
program loads it like any .dfm (models/dfm).

Workspace layout: dfm_workspace/<name>/{src,dst}/NNNNN.jpg + NNNNN_mask.png,
model.pt, preview.jpg, status.json, config.json, train.log, STOP (request).
"""

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKSPACES = os.path.join(PROJECT_DIR, "dfm_workspace")
