#!/bin/bash

inputs=$1 # "CA"

Njob=${2:-368} # default 368
Nproc=${3:-20} # default 20

function CMD {
	i=$1
	count=0
	for id_graph in `seq 0 $Njob`
	do
        count=$((count+1))
        if [ $count -eq $i ]
        then
            echo "Job $1 Ijob $2 start"
            python graph-task-C.py --id_graph=$id_graph --inputs=$inputs > log/"$inputs"_id_graph_$id_graph.txt
            echo $inputs id_graph $id_graph finished >> log/monitor_tmp.txt
            echo "Job $1 Ijob $2 ended"
        fi
	done
}

# do NOT change the follwing code
PID=()
for((i=1; i<=Njob; )); do
	for((Ijob=0; Ijob<Nproc; Ijob++)); do
		if [[ $i -gt $Njob ]]; then
			break;
		fi
		if [[ ! "${PID[Ijob]}" ]] || ! kill -0 ${PID[Ijob]} 2> /dev/null; then
			CMD $i $Ijob &
			PID[Ijob]=$!
			i=$((i+1))
		fi
	done
	sleep 0.01
done
wait


