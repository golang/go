//+build ignore

package main

// Test of dynamic function calls.
// No interfaces, so no runtime/reflect types.

func A1() {
	A2(0)
}

func A2(int) {} // not address-taken

func B() {} // unreachable

var (
	C = func(int) {}
	D = func(int) {}
)

func main() {
	A1()

	pfn := C
	pfn(0) // calls C and D but not A2 (same sig but not address-taken)
}

// WANT:
// Dynamic calls
//   main --> init$1
//   main --> init$2
// Reachable functions
//   A1
//   A2
//   init$1
//   init$2
// Reflect types
