package main

// Tests of 'freevars' query.
// See go.tools/guru/guru_test.go for explanation.
// See freevars.golden for expected query results.

// TODO(adonovan): it's hard to test this query in a single line of gofmt'd code.

type T struct {
	a, b int
}

type S struct {
	x int
	t T
}

func f(int) {}

func main() {
	type C int
	x := 1
	const exp = 6
	if y := 2; x+y+int(C(3)) != exp { // @freevars fv1 "if.*{"
		panic("expected 6")
	}

	var s S

	for x, y := range "foo" {
		println(s.x + s.t.a + s.t.b + x + int(y)) // @freevars fv2 "print.*y."
	}

	f(x) // @freevars fv3 "f.x."

loop: // @freevars fv-def-label "loop:"
	for {
		break loop // @freevars fv-ref-label "break loop"
	}
}
