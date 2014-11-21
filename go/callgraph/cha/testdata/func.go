//+build ignore

package main

// Test of dynamic function calls; no interfaces.

func A(int) {}

var (
	B = func(int) {}
	C = func(int) {}
)

func f() {
	pfn := B
	pfn(0) // calls A, B, C, even though A is not even address-taken
}

// WANT:
// Dynamic calls
//   f --> A
//   f --> init$1
//   f --> init$2
