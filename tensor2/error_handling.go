package tensor2

type tensorErrResponser struct {
	t *Tensor
}

type plainErrResponser struct{}

// RespErr makes it possible to handle error on library caller side.
// If a function/method is called via RespErr, error is returned if happened.
// If not, panic will be triggered on an error.
var RespErr = &plainErrResponser{}

// MustDo panics if the given err is not nil.
func MustDo(err error) {
	if err != nil {
		panic(err)
	}
}

// MustGet receives (T, error), then panics on error is non-nil.
// If error is nil, it returns the T.
func MustGet[T any](obj T, err error) T {
	if err != nil {
		panic(err)
	}

	return obj
}

// MustGet2 can receive (T1, T2, error), then do the same thing with MustGet.
func MustGet2[T1, T2 any](obj1 T1, obj2 T2, err error) (T1, T2) {
	if err != nil {
		panic(err)
	}

	return obj1, obj2
}
