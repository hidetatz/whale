package whale

import "github.com/hidetatz/whale/tensor"

type Optimizer interface {
	Optimize(v *Variable)
}

type SGD struct {
	learnRate *tensor.Tensor
}

func NewSGD(learnRate float64) *SGD {
	return &SGD{learnRate: tensor.Scalar(learnRate)}
}

func (s *SGD) Optimize(v *Variable) {
	newData := device.Sub(v.GetData(), device.Mul(v.GetGrad().GetData(), s.learnRate))
	v.SetData(newData)
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

func (s *MomentumSGD) Optimize(v *Variable) {
	if _, ok := s.velocities[v]; !ok {
		s.velocities[v] = tensor.ZerosLike(v.GetData())
	}
	velocity := s.velocities[v]
	velocity = device.Mul(velocity, s.momentum)
	velocity = device.Sub(velocity, device.Mul(s.learnRate, v.GetGrad().GetData()))
	v.SetData(device.Add(v.GetData(), velocity))
}
