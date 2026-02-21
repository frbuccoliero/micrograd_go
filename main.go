package main

import (
	"fmt"
	"micrograd_go/engine"
)

func main() {
	a := engine.NewValue(1.0)
	b := engine.NewValue(2.0)
    c := engine.Add(a,b)

	fmt.Println(a, b, c)
}