#!/bin/bash

if [ -n "$1" ]
then
	if [ -e "$1" ]
	then
		filename=$1
        	if [ "${filename##*.}" == "py" ]
		then
			python $filename
		else
			echo "$1 is not a python file, ${filename##*.}"
		fi
	else
		echo "$1 not a file"
	fi
else
        echo "nothing here"
fi
