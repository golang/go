package main

// Tests of method promotion logic.

type A struct{ magic int }

func (a A) x() {
	if a.magic != 1 {
		panic(a.magic)
	}
}
func (a *A) y() *A {
	return a
}

type B struct{ magic int }

func (b B) p() {
	if b.magic != 2 {
		panic(b.magic)
	}
}
func (b *B) q() {
	if b != theC.B {
		panic("oops")
	}
}

type I interface {
	f()
}

type impl struct{ magic int }

func (i impl) f() {
	if i.magic != 3 {
		panic("oops")
	}
}

type C struct {
	A
	*B
	I
}

func assert(cond bool) {
	if !cond {
		panic("failed")
	}
}

var theC = C{
	A: A{1},
	B: &B{2},
	I: impl{3},
}

func addr() *C {
	return &theC
}

func value() C {
	return theC
}

func main() {
	// address
	addr().x()
	if addr().y() != &theC.A {
		panic("oops")
	}
	addr().p()
	addr().q()
	addr().f()

	// addressable value
	var c C = value()
	c.x()
	if c.y() != &c.A {
		panic("oops")
	}
	c.p()
	c.q()
	c.f()

	// non-addressable value
	value().x()
	// value().y() // not in method set
	value().p()
	value().q()
	value().f()
}
