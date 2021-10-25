#  MSRCall
Explore Multi-scale Deep Structure for Basecalling Oxford Nanopore Sequences.

## Preparation

### Data folder
You can put your own test reads in fast5 format in the `dataset` folder and modify the path in script `run_0_preprocee_testsets.sh`  
In our paper, we used dataset from Ryan Wick et. al. that can be downloaded from this website:
https://doi.org/10.26180/5c5a5fa08bbee
### Required packages
ctcdecode: https://github.com/parlance/ctcdecode.git  
torch==1.2.0  
torchvision==0.2.2  
Pillow==6.0  
numpy  
pip  
statsmodels  
python-Levenshtein  
einops  
statsmodels  
h5py  
mummer  
minimap2  

## Run

### Preprocess test data
    bash ./run_0_preprocee_testsets.sh
The preprocessed .npy files are put in the `preprocess_test directory.
### Run basecalling
```angular2
python call.py -model exp_backup/MSRCall/MSRCall.chkpt -records_dir preprocessed_test/Acinetobacter_pittii_16_377_0801/ -output MSRCall_out
```
You can change the test data by replacing `Acinetobacter_pittii_16_377_0801` with your own filename.  
Basecalled results are stored in the `MSRCall_out` folder.
## Dataset reference:
    Wick, R. R., Judd, L. M., & Holt, K. E. (2019). Performance of neural network basecalling tools for Oxford Nanopore sequencing. Genome biology, 20(1), 1-10.

## Logs
docker run --gpus all --name=SACall -it -v /home/d05006/research/2019_0705_deepnano/:/workspace/2019_0705_deepnano/ pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel bash
docker start SACall
docker exec -it SACall bash
pip install python-Levenshtein
pip install einops
pip install statsmodels
pip install h5py
cd ctcdecode/
pip install .
cd ..
conda create --name SACall
source activate SACall
conda install -c bioconda minimap2
conda install -c bioconda mummer
apt update
apt install vim
