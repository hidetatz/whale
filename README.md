## Openblas support

whale internally supports openblas, currently for Matmul (32-bit sgemm) in tensor.
Right now it only supports Linux. To use openblas, make sure below are satisfied:

* openblas installed. See [Installation Guide](https://github.com/OpenMathLib/OpenBLAS/wiki/Installation-Guide#linux)
  * Header file       : /opt/OpenBLAS/include/cblas.h
  * Lib file directory: /opt/OpenBLAS/lib/
* Shared object path is specified at build time (for static link) or runtime (for dynamic link).
  * export `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/OpenBLAS/lib`
* Build go binary using build tag `blas`
  * `go build -tags blas`

If `-tags blas` is not specified, generic matmul implementation using pure Go is used.
If openblas is not usable from Go while `-tags blas` is specified, the build or execution of the program will just fail.

## Plotting

Plotting requires gnuplot installed.

```
sudo apt install gnuplot-x11
```
