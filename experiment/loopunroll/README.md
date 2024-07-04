Loop unrolling sample program.

## In C++

```shell
g++ -O3 experiment/loopunroll/main.cc
UNROLL=0 ./a.out # don't unroll
UNROLL=1 ./a.out # do unroll
```

```shell
$ g++ -O3 experiment/loopunroll/main.cc
$ UNROLL=0 ./a.out
unroll enabled: 0
elapsed time =       0.9112540 sec
$ UNROLL=1 ./a.out
unroll enabled: 1
elapsed time =       0.6401020 sec
```

## In Go

```shell
go build experiment/loopunroll/main.go
UNROLL=0 ./main # don't unroll
UNROLL=1 ./main # do unroll
```

```shell
$ go build experiment/loopunroll/main.go
$ UNROLL=0 ./main
unroll enabled: false
elapsed time = 5.882793136s
$ UNROLL=1 ./main
unroll enabled: true
elapsed time = 2.252102409s
```
