#! /bin/bash
FILE=/dev/stdin

if [[ -n $1 ]]; then FILE=$1; fi

cat $FILE | awk '{n+=1; global_acc=$7; acc_1=substr($10, 2); acc_2=$11; acc_3=$12; acc_4=substr($13, 1, length($13)-1); s1_global_acc+=global_acc; s2_global_acc+=(global_acc**2); s1_acc_1+=acc_1; s2_acc_1+=(acc_1**2); s1_acc_2+=acc_2; s2_acc_2+=(acc_2**2); s1_acc_3+=acc_3; s2_acc_3+=(acc_3**2); s1_acc_4+=acc_4; s2_acc_4+=(acc_4**2)} END {print "samples: " n "\nGLOBAL_ACC | mean ÷ stddev: " s1_global_acc/n " ÷ " sqrt(s2_global_acc/n - (s1_global_acc/n)**2) "\nACC_CLASS_1 | mean ÷ stddev: " s1_acc_1/n " ÷ " sqrt(s2_acc_1/n - (s1_acc_1/n)**2) "\nACC_CLASS_2 | mean ÷ stddev: " s1_acc_2/n " ÷ " sqrt(s2_acc_2/n - (s1_acc_2/n)**2) "\nACC_CLASS_3 | mean ÷ stddev: " s1_acc_3/n " ÷ " sqrt(s2_acc_3/n - (s1_acc_3/n)**2) "\nACC_CLASS_4 | mean ÷ stddev: " s1_acc_4/n " ÷ " sqrt(s2_acc_4/n - (s1_acc_4/n)**2)}'
