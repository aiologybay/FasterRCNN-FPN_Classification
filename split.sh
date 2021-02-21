#!/bin/bash
for i in `cat ./data/Main/test.txt | awk -F'/' '{print $NF}'`
do
	mv ./data/JPEGImages/${i} ./data/testjpg
	array=(${i//.jpg/.xml})
	for var in ${array[@]}
	do
		mv ./data/Annotations/${var} ./data/testanno
	done
done
