# The Neural Moving Average Model for Scalable Variational Inference of State Space Models

This is code for the [Neural Moving Average Model paper](https://arxiv.org/abs/1910.00879).

## Dependencies
- TensorFlow 1.8.0
- NumPy 1.14.5

## AR code

### Structure
The folder `dat` contains data generated by `AR_dat_gen`. This data is then used by `AR.py` which trains a model. Both of these python files can be called by `main.py`. `main.py` uses the hyperparameters defined in `hyperparameters.txt` to (a) generate synthetic data and then (b) fit the model to that synthetic data.

### Running the Experiments
After downloading the source, inference for the AR(1) process can be performed using the default values (as listed in the paper):
```
python main.py hyperparameters.txt
```
This uses TensorBoard to visualise training losses and parameter posteriors.  Data for this purpose is saved into a directory called 'train'. To visualise this in TensorBoard call (in your working directory):
```
tensorboard --logdir=train/
```
and navigate to `localhost:6006` in your browser.

The data-generation process and/or the inference proceudre can be modified by either editing the values in the hyperparameter text file or appending command line options. The command line options are detailed as follows:

<table>
<tr><th> Command Line Options </th></tr>

</td><td>

Option | Description
---------------|------------
`-T`, `-time`  | Total time
`-i`, `-impute`  | Observations every `impute` time step
`-t`, `-theta`  | Theta used for generation
`-x`, `xzero`  | Initial condition
`-o`, `-obs_std`  | Observation stdev
`-k`, `-kernel_len`  | Kernel length
`-b`, `-batch_dims`  | Batch dimensions
`-f`, `-feat_window`  | Feature window
`-repair`  | Print defaults to be put in txt file

</td></tr> </table>

## Other code

The code for the other experiments can be run using the `lotka_volterra_partial.py`, `fitz_nag_NVP.py` and `SV_dense.py` scripts.

