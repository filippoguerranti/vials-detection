#! /bin/bash
FILE=/dev/stdin

if [[ -n $1 ]]; then FILE=$1; fi

cat ${FILE}| \
grep '\[TEST\]\[end epoch\]' | \
awk '{mcc=substr($0, index($0, "mcc")+4, 6); n+=1; s1+=mcc; s2+=(mcc**2)} \
END \
{mean=s1/n; stddev=sqrt(s2/n - (s1/n)**2); print mean " ± " stddev ", runs:" n}'