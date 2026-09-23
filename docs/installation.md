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

The installers resolve the Conda Forge native packages, then install the tested
PyPI releases `ultralytics==8.3.203` and `sahi==0.12.6` with pip's `--no-deps` option.
Their runtime dependencies remain Conda Forge packages, so pip cannot replace
Torch or OpenCV. The installer then finishes with a runtime check. If more than
one Conda installation is present, pass the intended executable explicitly:

```powershell
.\tools\create_windows_conda_env.ps1 -Device gpu -CondaExecutable D:\anaconda3\Scripts\conda.exe
```

The installer uses Conda's `libmamba` solver and requires Conda 23.10 or newer.
If that solver fails on a supported installation, retry once with Conda's
documented fallback:

```powershell
.\tools\create_windows_conda_env.ps1 -Device gpu -Solver classic
```

The solve is run with strict Conda Forge channel priority in a temporary
isolated Conda profile, independent of channels listed in a user-level
`.condarc`; the configured Conda environment directory is retained.

### Clean CPU validation record

On 2026-09-23, a fresh `celfdrive-windows-cpu` environment was created from
`environment-cpu-windows.yml` using `D:\anaconda3\Scripts\conda.exe` (Conda
26.5.3), PowerShell's default `RemoteSigned` local-machine policy, and the
default `libmamba` solver:

```powershell
.\tools\create_windows_conda_env.ps1 -Device cpu -CondaExecutable D:\anaconda3\Scripts\conda.exe
.\tools\check_windows_conda_environment.ps1 -CondaExecutable D:\anaconda3\Scripts\conda.exe -EnvironmentName celfdrive-windows-cpu
```

The clean environment resolved Python 3.11.16, PyTorch 2.13.0 CPU, OpenCV
4.13.0, Pillow 12.3.0, tifffile 2024.2.12, `ultralytics==8.3.203`,
`ultralytics-thop==2.0.18`, and `sahi==0.12.6`. It passed the SlideBook runtime
check, the installation smoke test, and the complete test suite (248 passed;
two tifffile deprecation warnings). The exact Conda package URLs are recorded
in `Environments/celfdrive-windows-cpu.lock.txt`; recreate that validated native
stack with `conda create --name celfdrive-windows-cpu --file Environments/celfdrive-windows-cpu.lock.txt`, then run the two pinned pip installers above.

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
