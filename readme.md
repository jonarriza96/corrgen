# CorrGen

**CorrGen — A Differentiable Collision-Free Corridor Generator.**

|              Camera              |   Point cloud with corridors    |
| :------------------------------: | :-----------------------------: |
| ![camera](docs/kitti_camera.png) | ![corridors](docs/e2_kitti.gif) |

For the implementation details, please check the [paper](https://jonarriza96.github.io/#contact) (coming soon) and/or watch the [video](https://youtu.be/MvC7bPodXz8).

If you use this framework please cite our work:

```
Coming soon ...
```

## Quickstart

Install dependencies with

```
sudo apt-get install libcdd-dev
```

Create a python environment with python 3.9. For example, with conda:

```bash
conda create --name corrgen python=3.9
conda activate corrgen
pip install -r requirements.txt
```

Update the `~/.bashrc` with

```bash
export CORRGEN_PATH=/path_to_pfdq
export PYTHONPATH=$PYTHONPATH:/$CORRGEN_PATH
```

## Usage

### KITTI dataset

To run a real-world example from the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/raw_data.php) (Figs 5 and 6 in the paper), run this command:

```bash
python examples/kitti.py --case p  --lp --n_corrgen 6 --n_decomp 6
```

The options are the following ones:

- `--case`: `p` (pink corridor) or `g` (green corridor).
- `--lp`: Runs the approximated LP instead of the original SDP
- `--n_corrgen`: Integer indicating the polynomial degree of the polynomials in corrgen.
- `--n_decomp`: Sets the number of polygons for convex decomposition (and runs it)
<!-- - `--no_visualization`: Deactivates visualization
- `--save`: Saves the results. Make sure you update the path in the script. -->

### Toy example

To run a toy example (Fig. 4 in the paper), run this command:

```bash
python examples/toy_example.py --lp --n_corrgen 6
```

The options are the same as for the KITTI example (except for `--case`).

### 2D cross section comparison

To run the comparison of using different cross section parameterizations (Fig.3 in the paper), run this command:

```bash
python examples/cross_section.py
```

Notice that every time you run the command, the point cloud in the cross section varies. This is a great standalone script, great for conceptual prototyping.
