//go:build ignore
// +build ignore

package main

// Test of value flow from panic() to recover().
// We model them as stores/loads of a global location.
// We ignore concrete panic types originating from the runtime.

var someval int

type myPanic struct{}

func f(int) {}

func g() string { return "" }

func deadcode() {
	panic(123) // not reached
}

func main() {
	switch someval {
	case 0:
		panic("oops")
	case 1:
		panic(myPanic{})
	case 2:
		panic(f)
	case 3:
		panic(g)
	}
	ex := recover()
	print(ex)                 // @types myPanic | string | func(int) | func() string
	print(ex.(func(int)))     // @pointsto command-line-arguments.f
	print(ex.(func() string)) // @pointsto command-line-arguments.g
}
