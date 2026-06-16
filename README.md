# CelFDrive

An Artificial Intelligence assisted tool for automated microscopy.

## Installation

### Clone the repository:

```
git clone https://github.com/ScotfaAI/CelFDrive.git
```

### Create the conda Environment

```
conda env create -f environment-gpu-windows.yml --prefix Path/to/env
```

### Modifications to predict.py

Edit the global to set the absolute location of your cloned repository.

```
repo_path = "path\\to\\CelFDrive"
```

The primary function in predict.py is get_target_location it takes:

To modify this to work for a problem outside mitosis the get_target_location
function, the class_info dictionary needs to be modified to your experiment.

```
class_info = {
        class_id: (class_name, acceptable_confidence, priority_ranking),
        ...
        }
```

If you have another model you can replace Ultralytics with a pytorch model,
but you may need to modify the coordinate calculations as the current version
is based on normalised xywh coordinates where xy is the centre of the object.

## Defining a Workflow



## Training Data

CellClicker and CellSelector can be used to generate YOLO compatible labels from time series data.

### Requirements

Each dataset for training needs to contained in a single folder which has a subfolder called "images". Each file should end in t001.extension for a given timepoint.

### CellClicker

From inside your conda environment:

```
python run_clicker.py
```

### CellSelector

Edit run_selector.py and change this line to be all of your ordered classes.

```
phases = ['prophase','earlyprometaphase', 'prometaphase', 'metaphase', 'anaphase', 'telophase']
```

From inside your conda environment:

```
python run_selector.py
```

### Conversion

Edit the following variables in run_conversion.py

```
user = 'Scott'
imgpath = 'Path/to/Dataset'
```

From inside your conda environment:

```
python run_conversion.py
```

## Integration with software

This software is easy to deploy with intelligent-imaging-innovations Conditional Capture, but will work with imaging software that allows python code to be run such as LabVIEW and Micro-Manager.
