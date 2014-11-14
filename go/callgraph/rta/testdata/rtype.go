//+build ignore

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
// Dynamic calls
// Reachable functions
//   use
// Reflect types
//   *B
//   B
//   string
//   struct{uint64}
//   uint
//   uint64
