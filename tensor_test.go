package whale

import (
	"fmt"
	"testing"
)

func TestTensorString(t *testing.T) {
	ts := Ones(2, 3, 3)

	expected := `[
  [
    [1 1 1]
  ]
  [
    [1 1 1]
  ]
  [
    [1 1 1]
  ]
]
[
  [
    [1 1 1]
  ]
  [
    [1 1 1]
  ]
  [
    [1 1 1]
  ]
]
`

	if ts.String() != expected {
		t.Errorf("expected: '%s', got: '%s'", expected, ts.String())
	}
}
