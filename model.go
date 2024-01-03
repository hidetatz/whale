package whale

type Model interface {
	Train(in *Variable) (*Variable, error)
	Loss() LossCalculator
	Optimizer() Optimizer
	Params() []*Variable
}
