# pyNLControl

`pyNLCOntrol` is a package to solve general estimation and control problem (including non-linear problem). Further, it also  provides different method for analysis of dynamic system. This package is based on `CasADi` for python ([https://web.casadi.org/](https://web.casadi.org/)). This means problem should be formulated in `CasADi`. 

## Requirements
* python >= 3.6 (might work on older version of python3, not tested)
* casadi>=3.5.5 and jinja2>=3.0.2 `pip install casadi jinja2`


## Installations

```
pip install pyNLControl
```

## Supported control and estimator
* Estimators: Kalman filter, Extended Kalman Filter, Unscented Kalman Filter and simple Moving Horizon Estimators. Partical filter,  advanced moving horizon estimator, etc will be added soon.
* Control: LQR and simple Model Predictive Control. Other controllers will be added soon
* Misc: Nonlinear observability analysis, Noise covariance identification will be added soon