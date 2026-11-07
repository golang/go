// -lang=go1.27

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func id[T any](x T) T           { return x }
func pair[A, B any](a A, b B) B { return b }

type (
	F func(int) int
	P func(int, string) string
	S struct{ f F }
	T struct{ p P }
)

// Function type inference when assigning to a variable
// (incl. parameters and results) is permitted since Go 1.21.

func g(F)     {}
func gv(...F) {}
func gs(S)    {}
func h(P)     {}
func r1() F   { return id }
func r2() P   { return pair[int] }
func r3() S   { return S{f: id} }

func _() {
	var f F = id
	var p P = pair[int]
	f = id
	p = pair[int]
	var f1, f2 F = id, id
	f1, f2 = id, id
	g(id)
	h(pair[int])
	var s S
	s.f = id
	m := map[string]F{}
	m["k"] = id
	_, _, _, _, _ = f, p, f1, f2, s

	// append arguments are passed like function arguments
	var fs []F
	fs = append(fs, id)
	fs = append(fs, id, id)
	var ps []P
	ps = append(ps, pair[int])
}

// Function type inference in all other assignment contexts is permitted since Go 1.27.

func _() {
	_ = S{f: id}
	_ = S{id}
	_ = T{p: pair[int]}
	_ = []F{id}
	_ = [1]P{pair[int]}
	_ = map[string]F{"k": id}
	_ = map[string]P{"k": pair[int]}
	_ = []S{{f: id}}

	// literals nested in variable contexts
	var s S = S{f: id}
	gs(S{f: id})
	var f F = F(id)
	gv([]F{id}...)
	var fs []F
	fs = append(fs, []F{id}...)
	_, _, _ = s, f, fs

	ch := make(chan F, 1)
	ch <- id
	chp := make(chan P, 1)
	chp <- pair[int]
	select {
	case ch <- id:
	}

	_ = F(id)
	_ = P(pair[int])
}

// Function type inference when assigning to a variable of type
// parameter type is permitted since Go 1.27 (go.dev/issue/77439).
// Passing to a parameter of type parameter type is permitted
// since Go 1.21.

func _[G ~func(int) int]() G {
	var g G = id
	g = id
	k := func(G) {}
	k(id)
	k(g)
	return id
}

func _(f F) {
	switch f {
	case id /* ERROR "invalid case id in switch on f (func can only be compared to nil)" */ :
	}
}
