#! /bin/bash
FILE=/dev/stdin

if [[ -n $1 ]]; then FILE=$1; fi

echo "loss, f1, mcc, prec, recall" 

# cat $FILE | sed -n "/run: ${R}/,/\[TEST\]\[end epoch\]/p" | grep 'Epoch:' | grep 'TRAIN' | grep -v 'end epoch' | awk -v run=$R '{loss_match=index($0, "loss"); f1_match=index($0, "f1"); mcc_match=index($0, "mcc"); prec_match=index($0, "prec"); recall_match=index($0, "recall"); print(run ",", substr($0, loss_match+5, 10) ",", substr($0, f1_match+3, 6) ",", substr($0, mcc_match+4, 6) ",", substr($0, prec_match+5, 6) ",", substr($0, recall_match+7, 6))}'
cat ${FILE}| \
grep '\[TEST\]\[end epoch\]' | \
awk '{print(\
substr($0, index($0, "loss")+5, 10) ",", \
substr($0, index($0, "f1")+3, 6) ",", \
substr($0, index($0, "mcc")+4, 6) ",", \
substr($0, index($0, "prec")+5, 6) ",", \
substr($0, index($0, "recall")+7, 6))}'