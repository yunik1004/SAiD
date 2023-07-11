# SAiD: Blendshape-based Audio-Driven Speech Animation with Diffusion

## Basic Installation

Run the following command to install it as a pip module:

```bash
pip install .
```

If you are developing this repo or want to run the scripts, run instead:

```bash
pip install -e .[dev]
```

If there is an error related to pyrender, install additional packages as follows:

```bash
apt-get install libboost-dev libglfw3-dev libgles2-mesa-dev freeglut3-dev libosmesa6-dev libgl1-mesa-glx
```

## Script

The directory [script](script) contains the useful scripts such as preprocessing, training, testing, and evaluation.
Refer to [script/README.md](script/README.md) for the detailed information.

## Blender Addon

The directory [blender-addon](blender-addon) contains the blender addon that can be used for the visualization of the blendshape coefficients.
Refer to [blender-addon/README.md](blender-addon/README.md) for the detailed information.

## Dataset

The directory [data](data) contains some small size data.
Refer to [data/README.md](data/README.md) for the detailed information.

In addition, the large dataset (such as blendshape coefficients sequence) can be found on [LINK]().
