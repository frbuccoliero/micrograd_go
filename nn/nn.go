package nn

import (
	"math/rand"
	"micrograd_go/engine"
)

type Neuron struct {
	w []*engine.Value
	b *engine.Value
	nl string // non linearity: empty (none), "relu" or "tanh"
}

func NewNeuron (in int, nl string) *Neuron {
	w := make([]*engine.Value, in)
	for i := 0; i < in; i++ {
		w[i] = engine.NewValue(rand.Float64(), nil, "w")
	}
	b := engine.NewValue(rand.Float64(), nil, "b")
	return &Neuron{w:w, b:b, nl:nl}
}

func (a *Neuron) Activate (input []*engine.Value) *engine.Value {
	out := engine.NewValue(0.0, nil, "")

	// out = sum(w_i * x_i) + b
	for i, w := range a.w {
		out = out.Add(w.Mul(input[i]))
	}
	out.Add(a.b)

	// add nonlinearity
	if a.nl == "relu" {
		return out.ReLU()
	} else if a.nl == "tanh" {
		return out.Tanh()
	} else {
		return out
	}
}