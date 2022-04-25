# Anytime Capacity Expansion in Medical Residency Match by Monte Carlo Tree Search
Code for reproducing results in the paper "[Anytime Capacity Expansion in Medical Residency Match by Monte Carlo Tree Search](https://arxiv.org/abs/2202.06570)".
Note that Gurobi license is required to solve linear programming problems and mixed integer programming problems.

## About
This paper considers the capacity expansion problem in two-sided matchings, where the policymaker is allowed to allocate some extra seats as well as the standard seats.
In medical residency match, each hospital accepts a limited number of doctors.
Such capacity constraints are typically given in advance.
However, such exogenous constraints can compromise the welfare of the doctors; some popular hospitals inevitably dismiss some of their favorite doctors.
Meanwhile, it is often the case that the hospitals are also benefited to accept a few extra doctors.
To tackle the problem, we propose an anytime method that the upper confidence tree searches the space of capacity expansions, each of which has a resident-optimal stable assignment that the deferred acceptance method finds.
Constructing a good search tree representation significantly boosts the performance of the proposed method.
Our simulation shows that the proposed method identifies an almost optimal capacity expansion with a significantly smaller computational budget than exact methods based on mixed-integer programming.

## Installation
This code is written in Python 3.
To install the required dependencies, execute the following command:
```bash
$ pip install -r requirements.txt
```

### For Docker User
Build the container:
```bash
$ docker build -t capacity_expansion .
```
After build finished, run the container:
```bash
$ docker run -it capacity_expansion
```

## Run Experiments
In order to compare the proposed algorithm to the existing algorithms via synthetic data experiments (w/o hospital-wise limits), execute the following command:
```bash
$ python run_synthetic_experiment_wo_college_wise_budgets.py --num_students=1000 --num_colleges=15 --budget=30 --correlation=0.4 --num_trials=10
```
In this experiment, the following options can be specified:
* `--num_students`: Number of residents.
* `--num_colleges`: Number of hospitals.
* `--budget`: Number of expansion slots.
* `--correlation`: Correlation level of student preferences. The default value is `0.0`.
* `--num_trial`: Number of trials to run experiments. The default value is `10`.

To evaluate the algorithms via synthetic data experiments with hospital-wise limits, execute the following command:
```bash
$ python run_synthetic_experiment_with_college_wise_budgets.py --num_students=1000 --num_colleges=15 --budget=30 --correlation=0.4 --num_trials=10
``` 

To evaluate the algorithms via real-data experiments, execute the following command:
```bash
$ python run_real_data_experiment.py --budget=30 --num_trials=10
``` 