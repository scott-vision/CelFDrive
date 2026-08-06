"""Run the legacy phase-selection-to-YOLO conversion with local settings."""

from CellClicker.convert_selections_multiphase import convert_selections_multiphase
from glob import glob

# Change to correct user xml and path to dataset
user = 'Scott'
imgpath = 'Path/to/Dataset'
convert_selections_multiphase(imgpath+'/user_selections/'+user+'.xml', imgpath+'/images/cell_reigons.xml', imgpath+'/'+user+'_labels', user, imgpath)
