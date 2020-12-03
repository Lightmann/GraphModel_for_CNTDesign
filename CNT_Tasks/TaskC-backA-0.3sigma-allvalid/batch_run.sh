
date
./run-task-C.sh "CARS" 3 20
date
./run-task-C.sh "CRS" 63 20
date
./run-task-C.sh "CAS" 63 20
date
./run-task-C.sh "CAR" 104 20
date
./run-task-C.sh "CS" 265 20
date
./run-task-C.sh "CR" 480 20
date
./run-task-C.sh "CA" 480 20
date

# 
python graph-task-C-collect-scores.py --inputs="CARS"
python graph-task-C-collect-scores.py --inputs="CRS"
python graph-task-C-collect-scores.py --inputs="CAS"
python graph-task-C-collect-scores.py --inputs="CAR"
python graph-task-C-collect-scores.py --inputs="CS"
python graph-task-C-collect-scores.py --inputs="CR"
python graph-task-C-collect-scores.py --inputs="CA"
