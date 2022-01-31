#! /bin/bash
FILE=/dev/stdin

if [[ -n $1 ]]; then FILE=$1; fi

cat ${FILE}| \
grep '\[TEST\]\[end epoch\]' | \
awk '{mcc=substr($0, index($0, "mcc")+4, 6); n_mcc+=1; s1_mcc+=mcc; s2_mcc+=(mcc**2); \
f1=substr($0, index($0, "f1")+3, 6); n_f1+=1; s1_f1+=f1; s2_f1+=(f1**2); \
pr=substr($0, index($0, "prec")+5, 6); n_pr+=1; s1_pr+=pr; s2_pr+=(pr**2); \
re=substr($0, index($0, "recall")+7, 6); n_re+=1; s1_re+=re; s2_re+=(re**2)} \
END \
{mean_mcc=s1_mcc/n_mcc; stddev_mcc=sqrt(s2_mcc/n_mcc - mean_mcc**2); print "mcc: " mean_mcc " ± " stddev_mcc ", runs:" n_mcc; \
mean_f1=s1_f1/n_f1; stddev_f1=sqrt(s2_f1/n_f1 - mean_f1**2); print "f1: " mean_f1 " ± " stddev_f1 ", runs:" n_f1; \
mean_pr=s1_pr/n_pr; stddev_pr=sqrt(s2_pr/n_pr - mean_pr**2); print "pr: " mean_pr " ± " stddev_pr ", runs:" n_pr; \
mean_re=s1_re/n_re; stddev_re=sqrt(s2_re/n_re - mean_re**2); print "re: " mean_re " ± " stddev_re ", runs:" n_re} \'