package tensor2

func (t *Tensor) Bool(f func(f float64) bool) *Tensor {
	c := t.Copy()
	for i := range c.data {
		if f(c.data[i]) {
			c.data[i] = 1
		} else {
			c.data[i] = 0
		}
	}

	return c
}
