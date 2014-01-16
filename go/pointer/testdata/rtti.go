package main

// Regression test for oracle crash
// https://code.google.com/p/go/issues/detail?id=6605
//
// Using reflection, methods may be called on types that are not the
// operand of any ssa.MakeInterface instruction.  In this example,
// (Y).F is called by deriving the type Y from *Y.  Prior to the fix,
// no RTTI (or method set) for type Y was included in the program, so
// the F() call would crash.

import "reflect"

var a int

type X struct{}

func (X) F() *int {
	return &a
}

type I interface {
	F() *int
}

func main() {
	type Y struct{ X }
	print(reflect.Indirect(reflect.ValueOf(new(Y))).Interface().(I).F()) // @pointsto main.a
}
