package whale

type Model interface {
	Train(in *Variable) (*Variable, error)
	LossFn() LossCalculator
	Optimizer() Optimizer
	Params() []*Variable

	SaveGobFile(filename string) error
}
