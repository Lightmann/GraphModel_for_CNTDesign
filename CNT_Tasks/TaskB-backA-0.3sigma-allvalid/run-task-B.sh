#!/bin/bash

Njob=368
Nproc=15

function CMD {
	i=$1
	count=0
	for id_graph in `seq 0 $Njob`
	do
        count=$((count+1))
        if [ $count -eq $i ]
        then
            echo "Job $1 Ijob $2 start"
            python graph-task-B.py --id_graph=$id_graph > log/id_graph_$id_graph.txt
            echo id_graph $id_graph finished >> log/monitor_tmp.txt
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


