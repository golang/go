//+build ignore

package main

// Test of interface calls.

func use(interface{})

type A byte // instantiated but not a reflect type

func (A) f() {} // called directly
func (A) F() {} // unreachable

type B int // a reflect type

func (*B) f() {} // reachable via interface invoke
func (*B) F() {} // reachable: exported method of reflect type

type B2 int // a reflect type, and *B2 also

func (B2) f() {} // reachable via interface invoke
func (B2) g() {} // reachable: exported method of reflect type

type C string // not instantiated

func (C) f() {} // unreachable
func (C) F() {} // unreachable

type D uint // instantiated only in dead code

func (D) f() {} // unreachable
func (D) F() {} // unreachable

func main() {
	A(0).f()

	use(new(B))
	use(B2(0))

	var i interface {
		f()
	}
	i.f() // calls (*B).f, (*B2).f and (B2.f)

	live()
}

func live() {
	var j interface {
		f()
		g()
	}
	j.f() // calls (B2).f and (*B2).f but not (*B).f (no g method).
}

func dead() {
	use(D(0))
}

// WANT:
// Dynamic calls
//   live --> (*B2).f
//   live --> (B2).f
//   main --> (*B).f
//   main --> (*B2).f
//   main --> (B2).f
// Reachable functions
//   (*B).F
//   (*B).f
//   (*B2).f
//   (A).f
//   (B2).f
//   live
//   use
// Reflect types
//   *B
//   *B2
//   B
//   B2
