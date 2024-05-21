Whale is a deep learning framework.

Current project status:

* It can solve simple linear regression using multi layer perceptron. [example](https://github.com/hidetatz/whale/blob/master/examples/mlp/main.go)
* It can solve simple nonlinear regression (usually called "spiral dataset") using multi layer perceptron. [example](https://github.com/hidetatz/whale/blob/master/examples/spiral/main.go)
* It can solve mnist, 90% accuracy on testdata after 5 epoch training. Training is very slow (2min/1 epoch). [example](https://github.com/hidetatz/whale/blob/master/examples/spiral/main.go)
* CPU optimization is on-going.
* GPU support is future work.

## Plotting

Plotting requires gnuplot installed.

```
sudo apt install gnuplot-x11
```

## Openblas support

whale internally supports openblas through cgo, currently for Matmul (32-bit sgemm).
Right now it only supports Linux. To use openblas, make sure below are satisfied:

* openblas installed. See [Installation Guide](https://github.com/OpenMathLib/OpenBLAS/wiki/Installation-Guide#linux)
  * Header file must be named `/opt/OpenBLAS/include/cblas.h`
  * Lib files must locate in `/opt/OpenBLAS/lib/`
* Shared object path is specified at build time (for static link) or runtime (for dynamic link).
  * export `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/OpenBLAS/lib`
* Go program is built with build tag `blas`
  * `go build -tags blas`

If `-tags blas` is not specified, generic matmul implementation using pure Go (somewhat parallelized using goroutine) is used.
If openblas is not usable from Go while `-tags blas` is specified, the build or execution of the program will just fail.

