package main

import "./a"

func main() {
	v := a.New(7)
	if v == nil {
		panic("nil result")
	}
	if v.M() != 7 {
		panic("wrong result")
	}

	// The thunk still serves func values.
	fp := a.New
	if fp(9).M() != 9 {
		panic("func value")
	}

	// Error-shaped results keep their nil semantics.
	if err := a.Check(1); err != nil {
		panic("err should be nil")
	}
	if err := a.Check(-1); err == nil {
		panic("err should be non-nil")
	}
}
