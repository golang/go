package main

import (
	"./p"
)

type T struct{ *p.S }
type I interface {
	get()
}

func main() {
	var t T
	p.F(t)
	var x interface{} = t
	_, ok := x.(I)
	if ok {
		panic("should not satisfy main.I")
	}
	_, ok = x.(p.I)
	if !ok {
		panic("should satisfy p.I")
	}
}
