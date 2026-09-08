# Environment and reproducibility

The primary OmicPro environment is aligned with the tested `lmx-dcp_v2`
environment on `server3`: Linux, Python 3.10.20, PyTorch 2.5.1 with CUDA 12.1,
and Mamba 2.3.1. The verified server GPU is an NVIDIA RTX 4090.

## Create the core environment

```bash
conda env create -f 01OmicPro_code/environment.yml
conda activate omicpro
pip install --no-deps -e 01OmicPro_code
```

`requirements.txt` fixes the core model/data-science packages. It targets the
PyTorch CUDA 12.1 wheel index; use a compatible NVIDIA driver and Linux CUDA
host. Install optional figure tooling only when needed:

```bash
pip install -r 01OmicPro_code/requirements-figures.txt
```

## Use the existing server3 environment

On server3, Conda must be initialized in a non-interactive SSH shell before the
environment can be activated:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lmx-dcp_v2
export CUDA_VISIBLE_DEVICES=0
```

GPU 0 was verified with PyTorch. GPU 1 reported a driver/device-handle error
during inspection, so reproducibility commands should keep `CUDA_VISIBLE_DEVICES=0`.

Then validate and run the public synthetic demo:

```bash
python -m omicpro.check_data --config 01OmicPro_code/configs/demo.yaml
python -m omicpro.train_cv --config 01OmicPro_code/configs/demo.yaml --trait_name demo_trait --folds 1
```

## Lock-file policy

`requirements-server3.lock.txt` is a dated package snapshot of the server3
environment, including transitive packages. It is an audit/rebuild reference,
not the normal installer: use `environment.yml` and `requirements.txt` for new
OmicPro environments. Regenerate the lock after a deliberate server upgrade.
