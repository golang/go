// errorcheck -0 -d=ssa/opt/debug=3

package main

// Trivial interface call devirtualization test.

type real struct {
	value int
}

func (r *real) Value() int { return r.value }

type Valuer interface {
	Value() int
}

type indirectiface struct {
	a, b, c int
}

func (i indirectiface) Value() int {
	return i.a + i.b + i.c
}

func main() {
	var r Valuer
	rptr := &real{value: 3}
	r = rptr

	if r.Value() != 3 { // ERROR "de-virtualizing call$"
		panic("not 3")
	}

	// Can't do types that aren't "direct" interfaces (yet).
	r = indirectiface{3, 4, 5}
	if r.Value() != 12 {
		panic("not 12")
	}
}
