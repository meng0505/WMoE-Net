# WMoE-Net

WMoE-Net is a dual-stream oriented object detection framework for optical and SAR image fusion. It is built on Ultralytics YOLO and adds wavelet-directed cross-attention alignment with CIQ-MoE feature fusion.

## Main Components

- Dual-stream optical/SAR feature extraction
- WDCAlign for wavelet-domain cross-modal alignment
- CIQ-MoE for adaptive multi-expert fusion
- OBB detection head for oriented object detection
- Representative fusion and ablation YAML files under `ultralytics/cfg/models/fuse`

## Repository Layout

```text
WMoE-Net/
|-- train_dual.py
|-- val_dual.py
|-- infer_dual.py
|-- ultralytics/
|   |-- cfg/models/fuse/
|   |-- data/
|   |-- engine/
|   |-- models/
|   |-- nn/
|   `-- utils/
`-- README.md
```

## Data Layout

Prepare paired optical and SAR images with matching file names:

```text
datasets/
|-- images/
|   |-- train/
|   `-- val/
|-- imagesSAR/
|   |-- train/
|   `-- val/
`-- labels/
    |-- train/
    `-- val/
```

The dataset YAML used by the scripts is `datasets/ship.yaml`. Replace it with your own dataset file when needed.

## Dataset

The QXS-SAROPT-SHIP dataset is available at:

<p align="center">
  <img src="docs/assets/saropt_ship_preview.png" alt="QXS-SAROPT-SHIP optical and SAR examples" width="900">
</p>

- Zenodo: https://doi.org/10.5281/zenodo.20000667
- Baidu Disk: https://pan.baidu.com/s/1QbB70BxYquZ2drwubMnPpA

Baidu Disk extraction code: `fkz1fa`

## Training

```bash
python train_dual.py
```

The default model configuration is:

```text
ultralytics/cfg/models/fuse/wmoenet_wdca3_moe_top2.yaml
```

## Validation

```bash
python val_dual.py
```

By default this expects a local checkpoint at:

```text
weights/best.pt
```

## Inference

```bash
python infer_dual.py
```

Prediction outputs are written to `runs/predict/WMoE-Net`.

## Notes

Large files such as datasets, checkpoints, exported models, run outputs, and Python caches are intentionally excluded from version control.

## Acknowledgements

This project is built on top of the following open-source projects:

- YOLOFuse: https://github.com/WangQvQ/YOLOFuse
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics

We thank the original authors for releasing their code and providing a useful foundation for dual-stream multimodal object detection research.
