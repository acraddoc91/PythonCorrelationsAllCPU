#!/bin/bash

mount -t cifs //129.2.116.114/rbry /NAS -o user=sandy,password=105457Js
if [ -n "$1" ]
then
        if [ -e "$1" ]
        then
                filename=$1
                if [ "${filename##*.}" == "py" ]
                then
                        python $filename
                else
                        echo "$1 is not a python file"
                fi
        else
                echo "$1 not a file"
        fi
else
        echo "Nothing to do"
fi
