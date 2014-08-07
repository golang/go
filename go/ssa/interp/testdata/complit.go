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

func main() {
}
