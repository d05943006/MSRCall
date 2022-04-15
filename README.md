#  MSRCall
MSRCall: A Multi-scale Deep Neural Network for Basecalling Oxford Nanopore Sequences.

## Preparation

### Data folder
You can put your own test reads in fast5 format in the `dataset` folder and modify the path in script `run_0_preprocee_testsets.sh`  
In our paper, we used dataset from Ryan Wick et. al. that can be downloaded from this website:
https://doi.org/10.26180/5c5a5fa08bbee
### Required packages
Please install ctcdecode from:
ctcdecode: https://github.com/parlance/ctcdecode.git  
Install required python packages:
```angular2
pip install -r requirement.txt
```
optional software:
minimap2  

## Run

### Preprocess test data
    bash ./run_0_preprocee_testsets.sh
The preprocessed .npy files are put in the `preprocess_test` directory.
### Run basecalling
```angular2
python call.py -model exp_backup/MSRCall/MSRCall.chkpt -records_dir preprocessed_test/Acinetobacter_pittii_16_377_0801/ -output MSRCall_out
```
You can change the test data by replacing `Acinetobacter_pittii_16_377_0801` with your own filename.  
Basecalled results are stored in the `MSRCall_out` folder.
## Dataset reference:
    Wick, R. R., Judd, L. M., & Holt, K. E. (2019). Performance of neural network basecalling tools for Oxford Nanopore sequencing. Genome biology, 20(1), 1-10.
