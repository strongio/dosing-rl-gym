## Dosing-RL Gym

This repository contains a diabetic patient data simulator following the structure of an `open-ai gym`. It is intended to evaluate reinforcement learning algorithms.

For more information on how this gym can be used in applied reinforcement learning research, see [our blog post on personalized dosing](https://www.strong.io/blog/reinforcement-learning-for-personalized-medication-dosing).

## Installation

Install this package via `pip`:
```
python setup.py install
```

## Diabetic-v0 and Diabetic-v1

This simulator was inspired by the following work:
- [Maintain Glucose in Type-I Diabetic](http://apmonitor.com/pdc/index.php/Main/DiabeticBloodGlucose)
- [simglucose](https://github.com/jxx123/simglucose), an implementation of the FDA-approved 2008 version [UVA/Padova Simulator](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4454102/)

This simulator is based on an expanded version of the [Bergman minimal model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2769674/), which includes meal disturbances. The underlying mathematical representation of this model was first developed by [John D. Hedengren](http://apmonitor.com/pdc/index.php/Main/HomePage).

The goal is to keep glucose levels at a tolerable level in Type-1 diabetic patients. This process can be controlled using remote insulin uptake.

For additional details on this gym, see `dosing_rl_gym/resources/Diabetic Background.ipynb`.
