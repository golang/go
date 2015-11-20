package main

// Tests of composite literals.

import "fmt"

// Map literals.
func init() {
	type M map[int]int
	m1 := []*M{{1: 1}, &M{2: 2}}
	want := "map[1:1] map[2:2]"
	if got := fmt.Sprint(*m1[0], *m1[1]); got != want {
		panic(got)
	}
	m2 := []M{{1: 1}, M{2: 2}}
	if got := fmt.Sprint(m2[0], m2[1]); got != want {
		panic(got)
	}
}

// Nonliteral keys in composite literal.
func init() {
	const zero int = 1
	var v = []int{1 + zero: 42}
	if x := fmt.Sprint(v); x != "[0 0 42]" {
		panic(x)
	}
}

// Test for in-place initialization.
func init() {
	// struct
	type S struct {
		a, b int
	}
	s := S{1, 2}
	s = S{b: 3}
	if s.a != 0 {
		panic("s.a != 0")
	}
	if s.b != 3 {
		panic("s.b != 3")
	}
	s = S{}
	if s.a != 0 {
		panic("s.a != 0")
	}
	if s.b != 0 {
		panic("s.b != 0")
	}

	// array
	type A [4]int
	a := A{2, 4, 6, 8}
	a = A{1: 6, 2: 4}
	if a[0] != 0 {
		panic("a[0] != 0")
	}
	if a[1] != 6 {
		panic("a[1] != 6")
	}
	if a[2] != 4 {
		panic("a[2] != 4")
	}
	if a[3] != 0 {
		panic("a[3] != 0")
	}
	a = A{}
	if a[0] != 0 {
		panic("a[0] != 0")
	}
	if a[1] != 0 {
		panic("a[1] != 0")
	}
	if a[2] != 0 {
		panic("a[2] != 0")
	}
	if a[3] != 0 {
		panic("a[3] != 0")
	}
}

// Regression test for https://github.com/golang/go/issues/10127:
// composite literal clobbers destination before reading from it.
func init() {
	// map
	{
		type M map[string]int
		m := M{"x": 1, "y": 2}
		m = M{"x": m["y"], "y": m["x"]}
		if m["x"] != 2 || m["y"] != 1 {
			panic(fmt.Sprint(m))
		}

		n := M{"x": 3}
		m, n = M{"x": n["x"]}, M{"x": m["x"]} // parallel assignment
		if got := fmt.Sprint(m["x"], n["x"]); got != "3 2" {
			panic(got)
		}
	}

	// struct
	{
		type T struct{ x, y, z int }
		t := T{x: 1, y: 2, z: 3}

		t = T{x: t.y, y: t.z, z: t.x} // all fields
		if got := fmt.Sprint(t); got != "{2 3 1}" {
			panic(got)
		}

		t = T{x: t.y, y: t.z + 3} // not all fields
		if got := fmt.Sprint(t); got != "{3 4 0}" {
			panic(got)
		}

		u := T{x: 5, y: 6, z: 7}
		t, u = T{x: u.x}, T{x: t.x} // parallel assignment
		if got := fmt.Sprint(t, u); got != "{5 0 0} {3 0 0}" {
			panic(got)
		}
	}

	// array
	{
		a := [3]int{0: 1, 1: 2, 2: 3}

		a = [3]int{0: a[1], 1: a[2], 2: a[0]} //  all elements
		if got := fmt.Sprint(a); got != "[2 3 1]" {
			panic(got)
		}

		a = [3]int{0: a[1], 1: a[2] + 3} //  not all elements
		if got := fmt.Sprint(a); got != "[3 4 0]" {
			panic(got)
		}

		b := [3]int{0: 5, 1: 6, 2: 7}
		a, b = [3]int{0: b[0]}, [3]int{0: a[0]} // parallel assignment
		if got := fmt.Sprint(a, b); got != "[5 0 0] [3 0 0]" {
			panic(got)
		}
	}

	// slice
	{
		s := []int{0: 1, 1: 2, 2: 3}

		s = []int{0: s[1], 1: s[2], 2: s[0]} //  all elements
		if got := fmt.Sprint(s); got != "[2 3 1]" {
			panic(got)
		}

		s = []int{0: s[1], 1: s[2] + 3} //  not all elements
		if got := fmt.Sprint(s); got != "[3 4]" {
			panic(got)
		}

		t := []int{0: 5, 1: 6, 2: 7}
		s, t = []int{0: t[0]}, []int{0: s[0]} // parallel assignment
		if got := fmt.Sprint(s, t); got != "[5] [3]" {
			panic(got)
		}
	}
}

// Regression test for https://github.com/golang/go/issues/13341:
// within a map literal, if a key expression is a composite literal,
// Go 1.5 allows its type to be omitted.  An & operation may be implied.
func init() {
	type S struct{ x int }
	// same as map[*S]bool{&S{x: 1}: true}
	m := map[*S]bool{{x: 1}: true}
	for s := range m {
		if s.x != 1 {
			panic(s) // wrong key
		}
		return
	}
	panic("map is empty")
}

func main() {
}
