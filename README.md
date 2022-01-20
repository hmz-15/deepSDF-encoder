# deepSDF-encoder

This repo contains a complete implementation of a encoder-decoder model that takes in point cloud observations and represents fine-grained 3D geometry as implicit signed distance field. It is adapted from [DI-Fusion](https://github.com/huangjh-pub/di-fusion) and [DeepSDF](https://github.com/facebookresearch/DeepSDF).

## Installation
### 1. System Dependencies
- Ubuntu >= 18.04
- cuda >= 10.2
- cudnn 7
- Python 3.7

### 2. Python Packages
We recommend installing python dependencies in a virtual environment (python 3.7):
```bash
pip3 install -r requirements.txt
```

### 3. Dependencies for Data Generation
To generate training data yourself, please install the following dependencies:
- [CLI11](https://github.com/CLIUtils/CLI11)
- [Eigen-3.3.9](https://gitlab.com/libeigen/eigen/-/releases/3.3.9)
    - Use other versions may cause issues
- [Pangolin](https://github.com/stevenlovegrove/Pangolin)
    - Make sure you use the stable release version
    - May need to install [mpark/variant](https://github.com/mpark/variant)
- [Flann](https://github.com/flann-lib/flann)
    - Make sure cuda toolkit is correctly installed and configured
    - Enable BUILD_CUDA_LIB before running 'cmake ..'

The general process to install the packages:
``` bash
cd <your-lib-folder>
# 1. Download the package or use git clone, for example:
git clone https://github.com/CLIUtils/CLI11.git
# 2. Create build folder under package path, for example:
cd CLI11 && mkdir build && cd build
# 3. Run cmake
cmake ..
# 4. Run make and install to the system
make
make install
```

## Data Generation
First build the cuda-based mesh sampler:
``` bash
cd <this-repo>/sampler_cuda
mkdir build && cd build
cmake ..
make -j
```

Then run:
``` bash
# Enable headless rendering
export PANGOLIN_WINDOW_URI=headless://
python data_generator_sdf.py configs/data-cutting-shape.yaml --nproc 8
```

Note that you need to specify the input and output path in ```configs/data-cutting-shape.yaml```. 

## Training
```bash
python network_trainer_sdf.py configs/train-sdf.yaml [--options]
```
Please refer to ```configs/train-sdf.yaml``` for different running options. The code supports parallel training. 

You can also use tensorboard to monitor the training process:
```bash
tensorboard --logdir di-checkpoints/train_cutting/tensorboard
```
