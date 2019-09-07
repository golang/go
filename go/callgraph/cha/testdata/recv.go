// +build ignore

package main

type I interface {
	f()
}

type J interface {
	g()
}

type C int // C and *C implement I; *C implements J

func (C) f()
func (*C) g()

type D int // *D implements I and J

func (*D) f()
func (*D) g()

func f(i I) {
	i.f() // calls C, *C, *D
}

func g(j J) {
	j.g() // calls *C, *D
}

// WANT:
// Dynamic calls
//   f --> (*C).f
//   f --> (*D).f
//   f --> (C).f
//   g --> (*C).g
//   g --> (*D).g
