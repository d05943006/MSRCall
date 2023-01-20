#  MSRCall
[MSRCall: A Multi-scale Deep Neural Network for Basecalling Oxford Nanopore Sequences](https://doi.org/10.1093/bioinformatics/btac435)

Yang-Ming Yeh and Yi-Chang Lu
## Preparation

### Data folder
You can put your own test reads in fast5 format in the `dataset` folder and modify the path in script `run_0_preprocee_testsets.sh`  
In our paper, we used dataset from Ryan Wick et. al. that can be downloaded from this website:
https://doi.org/10.26180/5c5a5fa08bbee
### Required packages
Our code is tested on cuda 10.0, cudnn 7.6

Please install ctcdecode from:
https://github.com/parlance/ctcdecode.git  
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

## Train

### Reproduction
You can reproduce the training process by:
```angular2
python MSRCall_train.py -save_model ${modelName} -as ${train_set_dir}/signals/ -al ${train_set_dir}/labels/ -es ${val_set_dir}/signals/ -el ${val_set_dir}/labels/
```
As for the training/validation set generation, we follow the same procedures used in SACall.
Please refer to [scripts for training set in SACall](https://github.com/huangnengCSU/SACall-basecaller/blob/master/scripts/KP_generate_dataset.sh) and  [scripts for validation set in SACall](https://github.com/huangnengCSU/SACall-basecaller/blob/master/scripts/KP_generate_validation_dataset.sh).
## References:
    Wick, R. R., Judd, L. M., & Holt, K. E. (2019). Performance of neural network basecalling tools for Oxford Nanopore sequencing. Genome biology, 20(1), 1-10.
    Huang, N., Nie, F., Ni, P., Luo, F., & Wang, J. (2022). SACall: A Neural Network Basecaller for Oxford Nanopore Sequencing Data Based on Self-Attention Mechanism. IEEE/ACM transactions on computational biology and bioinformatics, 19(1), 614–623.
