package tensor2

type errResponser struct{}

// RespErr makes it possible to handle error on library caller side.
// If a function/method is called via RespErr, error is returned if happened.
// If not, panic will be triggered on an error.
var RespErr = &errResponser{}
