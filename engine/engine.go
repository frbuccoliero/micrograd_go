package engine

import (
	"fmt"
	"math"
)

type Value struct{
	data float64
	grad float64
	backward func()	
	prev []*Value
	Name string // making public so i can change it from outside if needed
}

func NewValue(data float64, prev []*Value, name string) *Value {
	return &Value{data: data, grad: 0.0, backward: func(){}, prev: prev, Name: name}
}

func (a *Value) Add(b *Value) *Value {
	out := NewValue(a.data + b.data, []*Value{a,b}, "+")
	out.backward = func () {
		a.grad += out.grad
		b.grad += out.grad
	}
	return out
}

func (a *Value) Mul(b *Value) *Value {
	out := NewValue(a.data * b.data, []*Value{a,b}, "*")
	out.backward = func () {
		a.grad += b.data * out.grad
		b.grad += a.data * out.grad
	}
	return out
}

func (a *Value) ReLu() *Value {
	var res float64
	if a.data > 0 {
		res = a.data
	} else {
		res = 0.0
	}

	out := NewValue(res, []*Value{a}, "relu")
	out.backward = func () {
		if out.data > 0 {
			a.grad += out.grad
		}
	}

	return out
}

func (a *Value) Tanh() *Value {
	out := NewValue(math.Tanh(a.data), []*Value{a}, "tanh")
	out.backward = func () {
		a.grad += (1 - math.Pow(out.data,2)) * out.grad
	}
	return out
}

// sub and div as "negative" operations of add and mul

func (a *Value) Negate() *Value {
	return a.Mul(NewValue(-1.0, nil, "-1"))
}

func (a *Value) Invert() *Value {
	return NewValue(1/a.data, *Value[]{a}, "^-1") // assuming no zero division
}

func (a *Value) Sub(b *Value) *Value{
	return a.Add(b.Negate())
}

func (a *Value) Div(b *Value) *Value {
	return a.Mul(b.Invert())
}

func (a *Value) Backward() {
	// Build topo sort
	topo := []*Value{}
	visited := make(map[*Value]bool) // looks like go has no hashset, we use a map

	var helper func((*Value)) 
	helper = func (a *Value) {
		if !visited[a] {
			visited[a] = true
			for _,p := range a.prev {
				helper(p)
			}
			topo = append(topo, a)
		}
	}
	
	helper(a)
	a.grad = 1.0

	// Work in reverse order
	for idx := len(topo) -1; idx >= 0; idx-- {
		topo[idx].backward()
	}
}


func (a *Value) Print() {
	fmt.Printf("%s [%p] Value(data=%v, grad=%v)\n", a.Name,a, a.data, a.grad)
}