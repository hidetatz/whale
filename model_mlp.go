package whale

import (
	"encoding/gob"
	"math"
	"os"

	"github.com/hidetatz/whale/tensor"
)

type MLP struct {
	Weights    []*Variable
	Biases     []*Variable
	Loss       LossCalculator
	Optim      Optimizer
	Activation Activation
}

func NewMLP(layers [][]int, bias bool, act Activation, loss LossCalculator, optim Optimizer) *MLP {
	mlp := &MLP{Loss: loss, Optim: optim, Activation: act}

	// init weights and biases
	for _, l := range layers {
		scale := math.Sqrt(1.0 / float64(l[1]))
		w := NewVar(tensor.RandNorm(l[0], l[1]).Mul(tensor.Scalar(scale)))
		mlp.Weights = append(mlp.Weights, w)

		if bias {
			b := NewVar(tensor.Zeros(l[1]))
			mlp.Biases = append(mlp.Biases, b)
		}
	}

	return mlp
}

func (m *MLP) Train(in *Variable) (*Variable, error) {
	var x, y *Variable
	var err error

	x = in

	for i := range m.Weights {
		w := m.Weights[i]
		var b *Variable
		if m.Biases != nil {
			b = m.Biases[i]
		}

		x, err = Linear(x, w, b)
		if err != nil {
			return nil, err
		}

		if i == len(m.Weights)-1 {
			y = x
			break
		}

		// do activation if not last layer
		x, err = m.Activation.Activate(x)
		if err != nil {
			return nil, err
		}
	}

	return y, nil
}

func (m *MLP) LossFn() LossCalculator {
	return m.Loss
}

func (m *MLP) Optimizer() Optimizer {
	return m.Optim
}

func (m *MLP) Params() []*Variable {
	return append(m.Weights, m.Biases...)
}

func (m *MLP) SaveGobFile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}

	return gob.NewEncoder(f).Encode(m)
}
