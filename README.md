# VoxelWiseSEM.jl

This package tries to support you in analysing MRI data voxelwise with structural equation models.

It provides functionality for

1. reshaping the data from BIDS into the correct format for analyzing it with SEM
2. clean the data (check / remove outliers etc.)
3. fit the models and extract the relevant information

## BIDS standard
### Folder hierarchy
project/subject/session/datatype

project:    name of the project
subject:    sub-<label> with unique labels for each participant
session:    ses-<label> with unique labels for each session
datatype:   only anat is allowed atm

#### Masks
ATM, just one mask for all participants is allowed.
