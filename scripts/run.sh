# !/bin/bash

ARCHITECTURE="sae32 cnn32 sae90 cnn90"
ENCODED_SIZE=(20 50 100 500)
BATCH_SIZE=(64 256)
OPTIMIZER="adam sgd"
LEARNING_RATE_ADAM=(1e-4 1e-3)
LEARNING_RATE_SGD=(1e-2 1e-1)
RECONSTRUCTION_WEIGHTS=(0.2 0.5 1.0)
EPOCHS=20
RUNS=5
GPU=1

DATETIME=$(date +'%a, %d-%b-%Y %H:%M:%S')

ODIR=results/
mkdir -p ${ODIR}

# set -x

for B in ${BATCH_SIZE[@]}; do
    for C in ${ENCODED_SIZE[@]}; do
        for O in ${OPTIMIZER[@]}; do
            if [[ $O == "adam" ]]; then
                LEARNING_RATE=("${LEARNING_RATE_ADAM[@]}")
            else
                LEARNING_RATE=("${LEARNING_RATE_SGD[@]}")
            fi
            for LR in ${LEARNING_RATE[@]}; do
                for A in ${ARCHITECTURE[@]}; do
                    if [[ ${A:0:3} == "sae" ]]; then
                        REC_WEIGHTS=("${RECONSTRUCTION_WEIGHTS[@]}")
                    else
                        REC_WEIGHTS=(0.0)
                    fi
                    for RW in ${REC_WEIGHTS[@]}; do
                        # Define an array to contain the arguments to pass
                        args=(
                            --arch=$A
                            --train_augmentation
                            --valid_augmentation
                            --epochs=$EPOCHS
                            --batch_size=$B
                            --encoded_size=$C
                            --optimizer=$O
                            --learning_rate=$LR
                            --reconstruction_weight=$RW 
                            --imgs_size=${A:3:2}
                            --gpu=$GPU
                            --print_freq=1
                            )
                        F=$ODIR/${A}-${C}.txt
                        echo -e "$DATETIME\n$A | $B | $C | $O | $LR | $RW" | tee -a $F
                        for R in $(seq 1 $RUNS); do
                            echo -e "run: $R"
                            python3 main.py data/imgs "${args[@]}"
                        done | tee -a $F
                        echo -e "\n" | tee -a $F
                    done
                done
            done
        done
    done
done
