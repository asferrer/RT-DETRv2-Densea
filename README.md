# RT-DETRv2 for Marine Debris Detection

![CUDA](https://img.shields.io/badge/CUDA-12.1.0-orange)
![CUDNN](https://img.shields.io/badge/CUDNN-8.7.0-orange)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
<a href="https://arxiv.org/abs/2407.17140">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2407.17140-red">
</a>
<a href="mailto:your.email@example.com">
    <img alt="email" src="https://img.shields.io/badge/contact_me-email-yellow">
</a>

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Marine Debris Detection Goals](#marine-debris-detection-goals)
- [Installation](#installation)
  - [Using DevContainer](#using-devcontainer)
  - [Using Conda & Pip](#using-conda--pip)
- [Models](#models)
- [Training Commands](#training-commands)
- [Testing Commands](#testing-commands)
- [Docker Setup](#docker-setup)
- [Contributing](#contributing)
- [License](#license)
- [Citing](#citing)
- [Contact](#contact)

## Project Overview

Project **RT-DETRv2-DenSea** utilizes **RT-DETRv2**, a real-time object detection model, for the identification and classification of **marine debris** in underwater environments. The model is optimized for detecting waste materials on the ocean floor, improving upon previous object detection frameworks by incorporating transformer-based enhancements for **speed and accuracy**.

This implementation is based on **RT-DETRv2** and leverages its strengths to handle the unique challenges of underwater detection, including **low visibility, diverse debris shapes, and complex marine environments**.

## Features

- **High-Speed Marine Debris Detection:** RT-DETRv2 enables real-time object detection in underwater conditions.
- **Transformer-Based Model:** Utilizes detection transformers for improved performance in occluded and low-light scenarios.
- **Dataset Support:** Compatible with COCO and custom marine debris datasets (e.g., CleanSea).
- **Lightweight & Efficient:** Optimized for deployment on embedded GPU devices such as NVIDIA Jetson and edge AI solutions.
- **Dockerized Environment:** Includes a complete Docker setup for streamlined execution.

## Marine Debris Detection Goals

This project aims to assist marine conservation efforts by:

- **Monitoring underwater pollution** through AI-driven detection.
- **Supporting ocean cleanup initiatives** with accurate classification of waste materials.
- **Providing researchers with robust detection tools** for marine debris tracking.

## Installation

### Using DevContainer

1. Clone the repository and navigate to the project directory.
  ```bash
  git clone https://github.com/asferrer/DenSea-RTDETR.git
  cd RT-DETRv2-DenSea
  ```
2. Ensure Docker and Docker Compose are installed on your system.
3. Build and run the DevContainer:
  ```bash
  docker-compose up -d
  ```

### Using Conda & Pip

1. Clone the repository:
  ```bash
  git clone https://github.com/asferrer/RT-DETRv2-DenSea.git
  cd RT-DETRv2-DenSea
  ```
2. Create a virtual environment with Conda:
  ```bash
  conda create -n DenSea python=3.10
  conda activate DenSea
  ```
3. Install dependencies:
  ```bash
  pip install --upgrade pip
  pip install torch torchvision
  pip install timm
  ```

## Models

Summary of supported models and their performance:

| Model | Input Shape | Dataset | AP (mAP@IoU50) | FPS (TensorRT FP16) |
|---|---|---|---|---|
| RT-DETRv2-S | 640 | COCO | **48.1** | 217 |
| RT-DETRv2-M | 640 | COCO | **49.9** | 161 |
| RT-DETRv2-L | 640 | COCO | **53.4** | 108 |
| RT-DETRv2-X | 640 | COCO | **54.3** | 74 |

## Training Commands

To train the model on marine debris datasets, use:
```bash
python tools/train.py -c /app/RT-DETR/configs/rtdetrv2/rtdetrv2_r18vd_120e_densea_v4.yml -d cuda --seed 21
```

## Evaluation Commands

```bash
python tools/eval.py -c /app/RT-DETR/configs/rtdetrv2/rtdetrv2_r18vd_120e_densea_v4.yml -m /app/RT-DETR/output/rtdetrv2_r50vd_densea_v4/best.pth -o evaluation/rtdetrv2_r50vd_densea_v4 -d cuda
```
## Testing Commands

Run inference on images, videos, or live webcam feed:
```bash
python demo.py --config-file configs/rtdetr_marine_debris.yaml --input test.jpg --opts MODEL.WEIGHTS models/rtdetrv2_marine_debris.pth
```

For video input:
```bash
python demo.py --config-file configs/rtdetr_marine_debris.yaml --video-input sample_video.mp4 --opts MODEL.WEIGHTS models/rtdetrv2_marine_debris.pth
```

For real-time webcam detection:
```bash
python demo.py --config-file configs/rtdetr_marine_debris.yaml --webcam --opts MODEL.WEIGHTS models/rtdetrv2_marine_debris.pth
```

## Docker Setup

### Dockerfile
The project includes a Dockerfile based on `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` for easy deployment.

### docker-compose.yml
The `docker-compose.yml` provides a DevContainer setup with:
- Volume mounting for project files.
- Port forwarding for visualization tools.
- GPU acceleration for deep learning inference.

## Contributing

We welcome contributions! Follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes with meaningful messages.
4. Open a pull request.

## License

This project is licensed under the **Apache 2.0 License**.

## Citing

If you use `RT-DETRv2` in your research or wish to refer to the results published here, please use the following BibTeX entries.

```BibTeX
@article{SANCHEZFERRER2023154,
      title = {An experimental study on marine debris location and recognition using object detection},
      author = {Alejandro Sánchez-Ferrer and Jose J. Valero-Mas and Antonio Javier Gallego and Jorge Calvo-Zaragoza},
      journal = {Pattern Recognition Letters},
      year = {2023},
      doi = {https://doi.org/10.1016/j.patrec.2022.12.019},
      url = {https://www.sciencedirect.com/science/article/pii/S0167865522003889},
}
```
```BibTeX
@InProceedings{10.1007/978-3-031-04881-4_49,
      title="The CleanSea Set: A Benchmark Corpus for Underwater Debris Detection and Recognition",
      author="S{\'a}nchez-Ferrer, Alejandro and Gallego, Antonio Javier and Valero-Mas, Jose J. and Calvo-Zaragoza, Jorge",
      booktitle="Pattern Recognition and Image Analysis",
      year="2022",
      publisher="Springer International Publishing",
}
```
```BibTeX
@misc{lv2024rtdetrv2,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer},
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140},
}
```

## Contact

For inquiries and support, contact:

**Project Lead:** Alejandro Sanchez Ferrer  
**Email:** asanc.tech@gmail.com  
**GitHub:** [asferrer](https://github.com/asferrer)