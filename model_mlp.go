package whale

import (
	"github.com/hidetatz/whale/tensor"
)

type MLP struct {
	weights    []*Variable
	biases     []*Variable
	loss       LossCalculator
	optim      Optimizer
	activation Activation
}

func NewMLP(layers [][]int, bias bool, act Activation, loss LossCalculator, optim Optimizer) *MLP {
	mlp := &MLP{loss: loss, optim: optim, activation: act}

	// init weights and biases
	for _, l := range layers {
		w := NewVar(tensor.Rand(l[0], l[1]))
		mlp.weights = append(mlp.weights, w)

		if bias {
			b := NewVar(tensor.Zeros(l[1]))
			mlp.biases = append(mlp.biases, b)
		}
	}

	return mlp
}

func (m *MLP) Train(in *Variable) (*Variable, error) {
	var x, y *Variable
	var err error

	x = in

	for i := range m.weights {
		w := m.weights[i]
		var b *Variable
		if m.biases != nil {
			b = m.biases[i]
		}

		x, err = Linear(x, w, b)
		if err != nil {
			return nil, err
		}

		if i == len(m.weights)-1 {
			y = x
			break
		}

		// do activation if not last layer
		x, err = m.activation.Activate(x)
		if err != nil {
			return nil, err
		}
	}

	return y, nil
}

func (m *MLP) Loss() LossCalculator {
	return m.loss
}

func (m *MLP) Optimizer() Optimizer {
	return m.optim
}

func (m *MLP) Params() []*Variable {
	return append(m.weights, m.biases...)
}
