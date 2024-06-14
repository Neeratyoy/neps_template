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

# Final Evaluation

To continue training the best-found configuration:
```bash
python best_model_eval.py --root_directory <path-to-neps-output> --output_path final_model/ 
```

In case the searcher was a multi-fidelity optimizer and the incumbent was early-stopped, can set a new budget to continue training:
```bash
python best_model_eval.py --root_directory <path-to-neps-output> --output_path final_model/ --max_budget 15
```

However, if using an LR schedule configured for a different max budget, this may not be the  correct way to retrain.

In that case, this is more appropriate:
```bash
python best_model_eval.py --root_directory <path-to-neps-output> --output_path final_model/  --max_budget 15 --evaluate_from_scratch
```