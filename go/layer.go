package whale

func Linear(x, w, b *Variable) (*Variable, error) {
	t, err := MatMul(x, w)
	if err != nil {
		return nil, err
	}

	if b == nil {
		return t, nil
	}

	y, err := Add(t, b)
	if err != nil {
		return nil, err
	}

	return y, nil
}
