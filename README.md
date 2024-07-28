Whale is a deep learning framework.

Current project status:

* It can solve simple linear regression using multi layer perceptron. [example](https://github.com/hidetatz/whale/blob/master/examples/mlp/main.go)
* It can solve simple nonlinear regression (usually called "spiral dataset") using multi layer perceptron. [example](https://github.com/hidetatz/whale/blob/master/examples/spiral/main.go)
* It can solve mnist, 90% accuracy on testdata after 5 epoch training. Training is very slow (1min/1 epoch). [example](https://github.com/hidetatz/whale/blob/master/examples/mnist/main.go)
* CPU optimization is on-going.
* GPU support is a future work.

## Development

### Dependencies

* OpenBLAS

Some parts of code required OpenBLAS installed in your system. See GitHub actions yaml file for the installation procedure.

* gnuplot

Plotting requires gnuplot installed.

```shell
sudo apt install gnuplot-x11
```

### Running test

Run `./test_unit`.

### Running performance test

Run `perf_sgemm`.
Note that this performance test looks at a percentage of theoretical FLOPS, so it does not use Go's builtin benchmarking feature.
