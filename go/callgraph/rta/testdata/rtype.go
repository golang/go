//go:build ignore
// +build ignore

package main

// Test of runtime types (types for which descriptors are needed).

func use(interface{})

type A byte // neither A nor byte are runtime types

type B struct{ x uint } // B and uint are runtime types, but not the struct

func main() {
	var x int // not a runtime type
	print(x)

	var y string // runtime type due to interface conversion
	use(y)

	use(struct{ uint64 }{}) // struct is a runtime type

	use(new(B)) // *B is a runtime type
}

// WANT:
//
//  reachable main
//  reachable use
//
// !rtype A
// !rtype struct{uint}
//  rtype *B
//  rtype B
//  rtype string
//  rtype struct{uint64}
//  rtype uint
//  rtype uint64
// !rtype int
