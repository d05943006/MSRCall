#!/bin/bash

## Preprocessing all 10 test sets
## usage:
# bash ./run_0_preprocee_testsets.sh 

declare -a testset_all=("Acinetobacter_pittii_16_377_0801" "Haemophilus_haemolyticus_M1C132_1" 
                        "Klebsiella_pneumoniae_INF032"     "Klebsiella_pneumoniae_INF042" 
                        "Klebsiella_pneumoniae_KSB2_1B"    "Klebsiella_pneumoniae_NUH29" 
                        "Serratia_marcescens_17_147_1671"  "Shigella_sonnei_2012_02037" 
                        "Staphylococcus_aureus_CAS38_02"   "Stenotrophomonas_maltophilia_17_G_0092_Kos")

arraylength=${#testset_all[@]}
mkdir preprocessed_test
for (( i=1; i<${arraylength}+1; i++ ));
do
    testName=${testset_all[$i-1]}
    echo " Preprocessing ${i}/${arraylength}, ${testName} ..."
    fast5Path="./dataset/7676174_testFast5/${testName}"

    python generate_dataset/inference_data.py -fast5 ${fast5Path} -records_dir preprocessed_test/${testName}/ -raw_len 2048 # > preprocess_${testName}.log

done
rm *fail*


echo "//..............................................."
echo "// All preprocessing finished!"
echo "// See the log above for more detail."
echo "// Continue to running model:"
echo "bash run_1_test_all.sh MSRCall"
