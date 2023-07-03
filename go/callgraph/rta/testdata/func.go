//go:build ignore
// +build ignore

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
//
//  edge main --dynamic function call--> init$1
//  edge main --dynamic function call--> init$2
//
//  reachable A1
//  reachable A2
//  reachable init$1
//  reachable init$2
// !reachable B
//  reachable main
