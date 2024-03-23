# CorrGen
**CorrGen — A Differentiable Collision-Free Corridor Generator.**

## Quickstart
Install dependencies

Create a python environment with python 3.9. For example, with conda:

```bash
conda create --name corrgen python=3.9
conda activate corrgen
pip install -r python/requirements.txt
```


Update the `~/.bashrc` with

```bash
export CORRGEN_PATH=/path_to_pfdq
export PYTHONPATH=$PYTHONPATH:/$CORRGEN_PATH
```

## Usage
To run the KITTI example in the paper, run this command:
```bash
python kitti/kitti.py --case 2  --n_corrgen 6
```

The options are the following ones:

- `--case_study`: `2` (pink corridor) or `3` (green corridor).
- `--lp`: Runs the approximated LP instead of the original SDP
- `--n_corrgen`: Integer indicating the polynomial degree of the polynomials in corrgen.
- `--n_decomp`: Sets the number of polygons for convex decomposition (and runs it)
- `--no_visualization`: Deactivates visualization
- `--save`: Saves the results in the `pfdq/results/data` folder. It is recommended not to trigger this, since you will overwrite the results of the paper.