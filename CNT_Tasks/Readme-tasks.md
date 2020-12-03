Here are the main tasks

#### 1. DataPrepare

Prepare dataset for later tasks: from the original experiment data to an organized dataset including attributions `C,A,R,T,S,Y`.

**Main columns**

* `C` is `content` -- (`P3HT vol (µL)	D1 vol (µL)	D2 vol (µL)	D6 vol (µL)	D8 vol (µL)`)
* `A` is `Absorbance`
* `R` is `ln(Absorption Ratio)`
* `T` is `ln(Thickness)`  -- unmeasured for many samples, since it is expensive
* `S` is `ln(Sheet Resistance)`
* `Y` is `ln(Conductivity)` calculated by `Sheet Resistance` and `Thickness` (`Y=const-S-T`)

Note: we use the logarithmic values in `R,T,S,Y`

The dataset is generated via batch BO with three targets (see the column `BO_Target`). 

The column`Run` is the BO step. Run1 is initial dataset.

For each `C`, there are about six droplets, numbered by `Droplet No.`.



#### 2. GraphPrepare

Generate graphs for later tasks: 

A. the simplest graphs that have no middle nodes

B. graphs from `C` to `Y` (`C` is the only input) 

C. graphs from multiple nodes (such as `C,R`) to `Y`



#### 3. Batch run tasks

```bash
make run # run the tasks
# then collect scores of graph models such as
python graph-task-A-collect-scores.py # scores will save in 'scores_simple.xlsx'
```

TaskA: evaluate the scores for the simplest graphs

TaskB: evaluate the scores for graphs from `C` to `Y` (`C` is the only input)

TaskC: evaluate the scores for graphs from multiple nodes (such as `C,R`) to `Y`


