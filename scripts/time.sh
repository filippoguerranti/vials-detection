#! /bin/bash
FILE=/dev/stdin

if [[ -n $1 ]]; then FILE=$1; fi

cat ${FILE}| \
grep -e '\[TRAIN\] Epoch: \[20\]\[end epoch\]' -e '\[TEST\]\[end epoch\]' | \
awk '{
    if ($1 == "[TRAIN]") 
        {train_s1+=$23; train_s2+=$23**2; train_n+=1;}
    else 
        {test_s1+=$21; test_s2+=$21**2; test_n+=1;} 
    }
    END {print "train: " train_s1/train_n " ± " sqrt(train_s2/train_n - (train_s1/train_n)**2);
        print "test: " test_s1/test_n " ± " sqrt(test_s2/test_n - (test_s1/test_n)**2)}'