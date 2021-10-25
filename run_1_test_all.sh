#!/bin/bash

## Run testing on all 10 test sets
## usage:
# bash ./run_1_test_all.sh MSRCall
modelName=$1 #"MSRCall"
chkptPath="exp_backup/${modelName}/${modelName}.chkpt"
######################################
## Testset information and symbol
######################################
declare -a testset_all=("Acinetobacter_pittii_16_377_0801" "Haemophilus_haemolyticus_M1C132_1" 
                        "Klebsiella_pneumoniae_INF032"     "Klebsiella_pneumoniae_INF042" 
                        "Klebsiella_pneumoniae_KSB2_1B"    "Klebsiella_pneumoniae_NUH29" 
                        "Serratia_marcescens_17_147_1671"  "Shigella_sonnei_2012_02037" 
                        "Staphylococcus_aureus_CAS38_02"   "Stenotrophomonas_maltophilia_17_G_0092_Kos")
declare -a testSymbol_all=("AP"        "HH"   "KP_INF032"  "KP_INF042"  "KP_KSB2" 
                           "KP_NUH29"  "SeM"   "ShS"         "StA"         "StM")
declare -a nTest_all=("4466" "7349" "15111" "11195" "16726" "15169" "16637" "23204" "11044" "16002")
arraylength=${#testset_all[@]}
echo "//........................................"
echo "// There're totally ${arraylength} data to be tested"
echo "//........................................"
######################################
## Call on all 10 testsets
######################################
for (( i=1; i<${arraylength}+1; i++ ));
do
    testName=${testset_all[$i-1]}
    testSymbol=${testSymbol_all[$i-1]}
    echo " Processing ${i}/${arraylength}, ${testName} ..."
    refPath="./dataset/7676135_reference/${testName}_reference.fasta"
    pafName="results/basecall_minimap2_${modelName}_read_alignment_${testSymbol}.paf"
    csvName="results/basecall_minimap2_${modelName}_read_data_${testSymbol}.csv"

    python call.py -model ${chkptPath} -records_dir preprocessed_test/${testName}/ -output MSRCall_out

    minimap2 -x map-ont -c ${refPath} MSRCall_out/call.fasta > ${pafName}

    python scripts/read_length_identity.py MSRCall_out/call.fasta ${pafName} > ${csvName}

    mv MSRCall_out/call.fasta results/pred_${modelName}_${testSymbol}.fasta
done


######################################
## Checking sum of line number
######################################
echo "All testing finished!"
echo "//..............................................."
echo "// Checking sum of line number in each pred file"
echo "//..............................................."
for (( i=1; i<${arraylength}+1; i++ ));
do
    testSymbol=${testSymbol_all[$i-1]}
    nTest=${nTest_all[$i-1]}
    nLine=$(wc -l < "adam_results/basecall_minimap2_${modelName}_read_data_${testSymbol}.csv")
    if [ "${nLine}" != "${nTest}" ]
    then
        echo "Prediction line number check sum fail, file:"
        echo "    adam_results/pred_${modelName}_${testSymbol}.fasta"
    fi
done
echo "//..............................................."
echo "// All check finished!"
echo "// See the log above for more detail."
echo "// Continue to assembly"
echo "bash run_2_assembly_all.sh ${modelName}"
