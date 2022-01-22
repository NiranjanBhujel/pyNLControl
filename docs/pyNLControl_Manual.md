# pyNLControl

pyNLControl is the package for nonlinear control and estimation. It is python based package. Almost all of the module will be applicable to linear system as well. Some might be applicable only to nonlinear or linear only.

## Requirements
* python >= 3.6 (might work on older version of python3, not tested)
* casadi>=3.5.5 and jinja2>=3.0.2 `pip install casadi jinja2 `


## Installations

```
pip install pyNLControl
```

## Supported control and estimator
* Estimators: Kalman filter, Extended Kalman Filter and Unscented Kalman Filter
* Control: Nonlinear Model Predictive Control will be added soon
* Misc: Nonlinear observability analysis, Noise covariance identification will be added soon

Module pyNLControl.BasicUtils
=============================

Functions
---------

    
`Gen_Code(func, filename, dir='/', mex=False, printhelp=False)`
:   Function to generate c code (casadi generated as well as interface) for casadi function.
    
    Args:
        func (casadi.Function): CasADi function for which code needs to be generated
        filename (str): Name of the generated code
        dir (str, optional): Directory where codes need to be generated. Defaults to '/'.
        mex (bool, optional): Option if mex is required. Defaults to False.
        printhelp (bool, optional): Option if information about input/output and its size are to be printed . If mex is False, sfunction help is also printed. Defaults to False.