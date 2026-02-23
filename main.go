package main

import (
    "fmt"
	"micrograd_go/engine"
    "micrograd_go/nn"
)

func main() {
    fmt.Println("--- TESTING ENGINE ---")

	a := engine.NewValue(1.0, nil, "a")
	b := engine.NewValue(2.0, nil, "b")
    
    c := a.Add(b)
    c.Name = "c"

    d := engine.NewValue(-2.0, nil, "d")

    e := c.Mul(d)
    e.Name = "e"

    f := engine.NewValue(3.0, nil, "f")

    g := e.Add(f)
    g.Name = "g"
    // g = (a+b)*d + f

    // h := g.ReLu()
    h := g.Tanh()

	all_vars := []*engine.Value{a,b,c,d,e,f,g,h}

    for _,v := range all_vars {
        v.Print()
    }

    h.Backward()

    fmt.Println("---Back pass---")

    // expected on g.Backward()
    // dg/dg = 1
    // df/dg = 1
    // de/dg = 1
    // dd/dg = dd/de * de/dg = c * 1 = a+b = 3
    // dc/dg = dc/de * de/dg = d * 1 = -2
    // da/dg = da/dc * dc/dg = 1 * -2 = -2
    // db/dg = db/dc * dc/dg = 1 * -2 = -2

    for _,v := range all_vars {
        v.Print()
    }

    fmt.Println("--- TESTING MLP ---")

    a = engine.NewValue(1.0, nil, "a")
	b = engine.NewValue(2.0, nil, "b")
    c = engine.NewValue(3.0, nil, "c")

    n := nn.NewNeuron(2, "tanh")
    out := n.Activate([]*engine.Value{a,b})
    out.Print()


}