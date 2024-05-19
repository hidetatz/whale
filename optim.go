package whale

import "github.com/hidetatz/whale/tensor"

type Optimizer interface {
	Optimize(v *Variable) error
}

type SGD struct {
	learnRate *tensor.Tensor
}

func NewSGD(learnRate float64) *SGD {
	return &SGD{learnRate: tensor.Scalar(learnRate)}
}

func (s *SGD) Optimize(v *Variable) error {
	delta := v.GetGrad().GetData().Mul(s.learnRate)
	newData := v.GetData().Sub(delta)

	v.SetData(newData)
	return nil
}

type MomentumSGD struct {
	learnRate  *tensor.Tensor
	momentum   *tensor.Tensor
	velocities map[*Variable]*tensor.Tensor
}

func NewMomentumSGD(learnRate, momentum float64) *MomentumSGD {
	return &MomentumSGD{
		learnRate:  tensor.Scalar(learnRate),
		momentum:   tensor.Scalar(momentum),
		velocities: make(map[*Variable]*tensor.Tensor),
	}
}

func (s *MomentumSGD) Optimize(v *Variable) error {
	if _, ok := s.velocities[v]; !ok {
		s.velocities[v] = tensor.ZerosLike(v.GetData())
	}
	velocity := s.velocities[v]
	velocity = velocity.Mul(s.momentum)

	delta := s.learnRate.Mul(v.GetGrad().GetData())

	velocity = velocity.Sub(delta)

	newv := v.GetData().Add(velocity)

	v.SetData(newv)
	return nil
}
