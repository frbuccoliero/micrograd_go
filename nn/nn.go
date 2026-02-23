package nn

import (
	"math/rand"
	"micrograd_go/engine"
)

// NEURON

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

// LAYER

type Layer struct {
	in int
	out int
	neurons []*Neuron
}

func NewLayer(in int, out int, nl string) *Layer {
	neurons := make([]*Neuron, out)
	for i := 0; i < out; i++ {
		neurons[i] = NewNeuron(in,nl)
	}
	return &Layer{in:in, out:out, neurons: neurons}
}

func (a *Layer) Activate (input []*engine.Value) []*engine.Value {
	out := make([]*engine.Value, a.out)
	for i := 0; i < a.out; i++ {
		out[i] = a.neurons[i].Activate(input)
	}
	return out
}

// MLP

type MLP struct {
	layers []*Layer
}

func NewMLP(in int, layersSize []int, nl string) *MLP {
	layers := make([]*Layer, len(layersSize))

	prevSize := in

	for i, size := range layersSize {
		nlLayer := nl
		if i == len(layersSize)-1 {
			nlLayer = "" // no activation on last layer
		}

		layers[i] = NewLayer(prevSize, size, nlLayer)
		prevSize = size
	}

	return &MLP{layers: layers}
}

func (a *MLP) Activate(input []*engine.Value) []*engine.Value {
	for _, layer := range a.layers{
		input = layer.Activate(input)
	}
	return input
}