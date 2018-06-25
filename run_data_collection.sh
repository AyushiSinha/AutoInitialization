#!/bin/bash

i=0

# iterate over the text file
FILES="dataset/pose/2*"
for f in $FILES
do
	if [ "$i" -eq "0" ]
	then
		prev_f="$f"
	else
		curr_f="$f"
		diff=`echo - | paste $prev_f $curr_f | awk '{sum += ($3 - $1) * ($3 - $1)} END {print sqrt(sum)}'`
		echo $diff
		for j in $(seq 0 $diff)
		do
			#echo $j
			python collect_data.py $prev_f $curr_f $j 6
			python collect_data.py $prev_f $curr_f $j 7
			python collect_data.py $prev_f $curr_f $j 8
			python collect_data.py $prev_f $curr_f $j 9
			python collect_data.py $prev_f $curr_f $j 10
		done
		prev_f="$f"
	fi
	i=$(expr $i + 1)
done
