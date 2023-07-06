# LSEMINK

This repository contains a Matlab implementation of LSEMINK, a modified Newton-Krylov method introduced in

```
@article{Kan2023LSEMINK,
  title={LSEMINK: A Modified Newton-Krylov Method for Log-Sum-Exp
Minimization},
  author={Kelvin, Kan and Nagy, James, and Ruthotto, Lars},
  journal={arXiv preprint arXiv:2306.xxxx},
  year={2023}
}
```

# Getting Started

See [minimalExample.m](minimalExmple.m)

# Reproduce Results in the Paper

To run the experiments, we have five driver codes (four for multinomial logistic regression and one for geometric programming):

	1) Main_small_MNIST.m
	2) Main_small_CIFAR10.m
	3) Main_RFM_MNIST.m
	4) Main_AlexNet_CIFAR10.m
	5) Main_GeometricProgramming.m

You will also need to install Meganet and CVX.

# Dependencies

The LSEMINK method is developed with MATLAB R2022a and should run with newer versions as well. No MATLAB toolboxes are required.

The following packages are required to repeat the image classification experiments:
1) MATLAB’s deep neural network toolbox (this is for the transfer learning component in the CIFAR-10 experiments).
2) [Meganet](https://github.com/XtractOpen/Meganet.m)

To reproduce the geometric programming experiments, you will need [CVX](http://cvxr.com/cvx/download/).

# Acknowledgments

This material is in part based upon work supported by the US National Science Foundation Grants DMS-1751636 and DMS-2038118, the US AFOSR Grants 20RT0237 and FA9550-18-1-0167, AFOSR MURI FA9550-18-1-050, and ONR Grant No. N00014-18-1- 2527. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the funding agencies.
