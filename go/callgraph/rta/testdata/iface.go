//go:build ignore
// +build ignore

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
//
//  edge live --dynamic method call--> (*B2).f
//  edge live --dynamic method call--> (B2).f
//  edge main --dynamic method call--> (*B).f
//  edge main --dynamic method call--> (*B2).f
//  edge main --dynamic method call--> (B2).f
//
//  reachable (A).f
// !reachable (A).F
//  reachable (*B).f
//  reachable (*B).F
//  reachable (B2).f
// !reachable (B2).g
//  reachable (*B2).f
// !reachable (*B2).g
// !reachable (C).f
// !reachable (C).F
// !reachable (D).f
// !reachable (D).F
//  reachable main
//  reachable live
//  reachable use
// !reachable dead
//
// !rtype A
//  rtype *B
//  rtype *B2
//  rtype B
//  rtype B2
// !rtype C
// !rtype D
