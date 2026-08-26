# Installation and verification

This guide covers the supported CelFDrive environments and checks that should
be completed before connecting a microscope.

## Windows Conda

From the repository root, create one environment for the target computer:

```powershell
# NVIDIA GPU
.\tools\create_windows_conda_env.ps1 -Device gpu

# CPU-only computer
.\tools\create_windows_conda_env.ps1 -Device cpu
```

The GPU environment is named `celfdrive-windows`; the CPU environment is
`celfdrive-windows-cpu`. To activate one interactively, first run
`conda init powershell`, open a new PowerShell window, then use
`conda activate <environment-name>`.

The installers resolve the Conda Forge packages, install SAHI without allowing
pip to replace Conda packages, and finish with a runtime check. If more than
one Conda installation is present, pass the intended executable explicitly:

```powershell
.\tools\create_windows_conda_env.ps1 -Device gpu -CondaExecutable D:\anaconda3\Scripts\conda.exe
```

## Verify an installation

Run these commands from the repository root with the intended environment
active. Use `--device gpu` only for the GPU environment.

```powershell
python -m pytest -q
python examples/run_smoke_test.py
python tools/verify_slidebook_runtime.py --device cpu
```

The test suite requires no microscope hardware. The smoke test uses the
synthetic blank-image fixture only to check installation and deterministic
inference; it is not a biological benchmark. A separate versioned
CellCognition P0037 image-and-label fixture is included for exported-label
benchmark reproduction; see [benchmarking](benchmarking.md).

`tools/verify_slidebook_runtime.py` imports the inference libraries in the
same order as a SlideBook callback, then runs the bundled model on a tutorial
TIFF. This must pass before configuring hardware capture.

## Windows pip virtual environment

If Conda is unsuitable, create a separate pip virtual environment. Do not
install these packages into a Conda environment.

```powershell
.\tools\create_windows_venv.ps1 -Device gpu -CudaWheel cu118

# CPU-only computer
.\tools\create_windows_venv.ps1 -Device cpu
```

The creator uses Python 3.11, installs the selected PyTorch wheel first, then
installs the remaining dependencies and runs the same runtime verification.
`cu118` is the default; use `cu126` or `cu128` only when supported by the NVIDIA
driver. The default directories are `.venv-celfdrive-gpu` and
`.venv-celfdrive-cpu`.

## macOS

Create the macOS environment, activate it, then install SAHI:

```bash
conda env create --file Environments/environment-gpu-mac.yml
conda activate celfdrive-macos
python tools/install_sahi.py
```

## Maintaining environments

The environment files specify compatible package ranges rather than a single
lockfile. On a machine with a working environment, archive an exact lockfile
with:

```powershell
.\tools\export_environment_lock.ps1 -Name celfdrive-windows
```

Do not add pip OpenCV packages or set `KMP_DUPLICATE_LIB_OK=TRUE`; both can
hide an unstable native-library configuration in SlideBook.
