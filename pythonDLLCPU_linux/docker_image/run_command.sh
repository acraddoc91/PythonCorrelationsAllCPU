#!/bin/bash
echo "running"
mount -t cifs //129.2.116.114/rbry /NAS -o credentials=/home/rbry/auth/NAS_auth
if [ -n "$1" ]
then
        if [ "$1" == "command" ]
        then
                echo "Processing..."
                python /home/rbry/dummy_processor.py "$@"
        elif [ "$1" == "file" ]
        then
                echo "Running file"
        fi
else
        echo "Nothing to do"
fi
