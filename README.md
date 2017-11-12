# SGIMC

This is a python/cython implementation of the **S**parse **G**roup **I**nductive
**M**atrix **C**ompletion, a matrix completion algorithm, which utilizes side channel
features to infer missing entries in the target sparse matrix.

The main advantage of SGIMC is the built-in model selection capability both on
individual feature and group level. 


## Installation

To install the module just run
```sh
make install
```
This will compile and install the `sgimc` python package.
