# !/bin/bash

MODEL="sae32 cnn32 sae90 cnn90"
RUNS=1
EPOCHS=10
LR=1e-4
ENCODED_SIZE=(20 50 100)
BATCH_SIZE=256
# AUGMENTATION=(True False)
# TEST_AUGMENTATION=(True False)
AUGMENTATION=(True)
TEST_AUGMENTATION=(True)


ODIR=vials-detection/results/

# set -x

for B in $BATCH_SIZE; do
    for C in $ENCODED_SIZE; do
        for M in $MODEL; do
            mkdir -p ${ODIR}/${M}
            if [[ $M == "SAE" ]]; then
                RECONSTRUCTION_WEIGHTS="0.5 1.0"
            else
                RECONSTRUCTION_WEIGHTS="0.0"
            fi
            for RW in $RECONSTRUCTION_WEIGHTS; do
                for A in ${AUGMENTATION[@]}; do
                    for TA in ${TEST_AUGMENTATION[@]}; do
                        # Define an array to contain the arguments to pass
                        args=(
                            --epochs=$EPOCHS
                            --batch_size=$B
                            --encoded_size=$C
                            --learning_rate=$LR
                            --num_workers="12"
                            --device="cuda:1"
                            --datetime=$DATE
                            --reconstruction_weight=$RW 
                            )
                        if [ $A = True ]; then
                            args+=(--augmentation)
                            if [ $TA = True ]; then
                                args+=(--test_augmentation)
                            fi
                        else
                            if [ $TA = True ]; then
                                continue
                            fi
                        fi
                        F=$ODIR/${M}/C${C}_RW${RW}_B${B}_E${EPOCHS}_R${RUNS}_A${A}_TA${TA}_${DATE}.txt
                        for R in $(seq 1 $RUNS); do
                            echo "M $M | C ${C} | RW ${RW} | B ${B} | E ${EPOCHS} | R ${RUNS} | A${A} | TA${TA}" | tee -a $F
                            echo -e "\nR $R"
                            python3 vials-detection/vials-detection.py $M vials-detection/images "${args[@]}"
                        done | tee -a $F
                        echo -e "\n" | tee -a $F
                    done
                done
            done
        done
    done
done
