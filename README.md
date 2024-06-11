# Environment setup

```bash
conda create -n neps_template python=3.11   # this Python version is not a strict requirement

# pip install -r requirements.txt  # this will not work until neps>0.12.0, until then
# installing from main branch
pip install git+https://github.com/automl/neps.git
```

# Running code

To run `random search`:
```bash
python hpo_target.py --algo rs
```

To run `BO`:
```bash
python hpo_target.py --algo bo
```


To run `HyperBand`:
```bash
python hpo_target.py --algo hyperband
```


To run `PriorBand`:
```bash
python hpo_target.py --algo priorband
```

## Visualizations

```bash
tensorboard --logdir ./neps_output/<dir-name>  # check the yaml for each run's root directory
```