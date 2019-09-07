// +build ignore

package main

// Test of interface calls.  None of the concrete types are ever
// instantiated or converted to interfaces.

type I interface {
	f()
}

type J interface {
	f()
	g()
}

type C int // implements I

func (*C) f()

type D int // implements I and J

func (*D) f()
func (*D) g()

func one(i I, j J) {
	i.f() // calls *C and *D
}

func two(i I, j J) {
	j.f() // calls *D (but not *C, even though it defines method f)
}

func three(i I, j J) {
	j.g() // calls *D
}

func four(i I, j J) {
	Jf := J.f
	if unknown {
		Jf = nil // suppress SSA constant propagation
	}
	Jf(nil) // calls *D
}

func five(i I, j J) {
	jf := j.f
	if unknown {
		jf = nil // suppress SSA constant propagation
	}
	jf() // calls *D
}

var unknown bool

// WANT:
// Dynamic calls
//   (J).f$bound --> (*D).f
//   (J).f$thunk --> (*D).f
//   five --> (J).f$bound
//   four --> (J).f$thunk
//   one --> (*C).f
//   one --> (*D).f
//   three --> (*D).g
//   two --> (*D).f
