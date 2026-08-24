# CelFDrive SlideBook direct-Python setup

This workflow passes the raw SlideBook montage directly to Python, returns
CelFDrive target locations, switches to the high-resolution objective, and
runs the returned 6D capture. It does not use MATLAB.

## Before you start

- Install CelFDrive on the acquisition computer and create the Windows Conda
  environment from the repository root:

  ```powershell
  conda env create -f Environments/environment-gpu-windows.yml
  conda activate celfdrive-windows
  ```

- Configure SlideBook's Python integration to use that Conda environment. The
  environment must be registered in SlideBook under the same name used by
  `Python_SetEnvironment`; merely creating it with Conda is not sufficient.
- In the SlideBook Python console or a temporary test script, verify that the
  selected interpreter imports `numpy`, `ultralytics`, and `predict` after the
  CelFDrive checkout is added to `sys.path`.
- Confirm the target computer's SlideBook version provides
  `Python_SetEnvironment`, `Python_RunCommand`, and
  `Python_RunHierarchicalCaptureFunction`.

## Install and customise the scripts

1. Copy [CelFDrive.sbs](../SlideBook/CelFDrive.sbs) and
   [find_locations_of_interest_montage.py](../SlideBook/find_locations_of_interest_montage.py)
   to SlideBook's scripts folder. Open SlideBook's **Scripting** ribbon and
   choose **Open Scripts Folder** to locate it.
2. Run `python run_config_gui.py`. In **Coordinates > SlideBook Python Macro**,
   enter the environment name registered in SlideBook and the high-resolution
   objective. Optionally enter **Objective before target search** to add a
   `ChangeObjective(...)` command immediately before target finding. **Save**
   generates `SlideBook/CelFDrive.sbs` with those settings and the configured
   CelFDrive project root.
3. Copy the generated `CelFDrive.sbs` into SlideBook's scripts folder and
   restart SlideBook so it discovers the updated script. The Python bridge
   imports `predict` from the configured checkout; do not add a MATLAB path.

The supplied macro intentionally does not create a maximum-Z projection. The
bridge expects the raw montage image in SlideBook's
`(position, height, width)` order and converts it to CelFDrive's
`(height, width, position)` order. Do not add a projection command unless the
bridge is deliberately changed and retested for a two-dimensional input.

When `logging.timing.enabled` is enabled (the default), each successful
target-selection callback writes one timing line to the Python output:
preprocessing, inference, postprocessing, logging, and total seconds. Timings
accumulate across all montage positions; inference timing waits for CUDA work
to complete when a CUDA device is active. `logging.enabled` independently
controls creation of the logging directory, experiment folders, and annotated
images.

## Configure CelFDrive and SlideBook capture scripts

Run `python run_config_gui.py` and use the **Coordinates** tab to enter the
measured high-resolution-objective minus prescan-objective X, Y, and Z stage
offsets in micrometres. Leave all three at `0.0` only after verifying that the
objectives are parfocal and XY-registered. CelFDrive adds these offsets to
every returned target, including the no-detection location.

In **High Resolution Imaging**, set `profile.highres_script` to the exact name
of the SlideBook postscan script. Create that script with the desired channels,
exposure, Z range, time points, and multipoint behaviour. Also create the
script named by `no_detection.empty_3i_capture_script` (default: `donothing`)
so a no-detection result completes safely.

Configure the overview prescan as a montage and set its image name to
`CelFDrive`. In the capture window, set **Advanced > Conditional Capture >
After Capture > Run script** to `CelFDrive.sbs`.

## Validate before experimental use

1. Run a non-critical montage with one known target.
2. Confirm the raw image has one plane per returned stage position;
   an axis/position mismatch is an integration error, not a value to guess.
3. Confirm the target coordinates, objective offsets, generated postscan
   script, and `Run6DCapture()` output in SlideBook.
4. Run a blank montage and confirm that `donothing` is selected.

> **Important:** Stage directions, raw-image axis order, objective offsets, and
> capture limits are microscope-specific. Validate them on the acquisition
> system before collecting experimental data.
