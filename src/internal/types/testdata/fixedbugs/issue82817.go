// -lang=go1.26

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
func r3() S {
	return S{f: id /* ERROR "implicitly instantiated function in struct literal requires go1.27 or later" */}
}

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

// Function type inference in all other assignment contexts requires Go 1.27.

func _() {
	_ = S{f: id /* ERROR "implicitly instantiated function in struct literal requires go1.27 or later" */}
	_ = S{id /* ERROR "implicitly instantiated function in struct literal requires go1.27 or later" */}
	_ = T{p: pair[ /* ERROR "partially instantiated function in struct field requires go1.27 or later" */ int]}
	_ = []F{id /* ERROR "implicitly instantiated function in array or slice literal requires go1.27 or later" */}
	_ = [1]P{pair[ /* ERROR "partially instantiated function in array or slice element requires go1.27 or later" */ int]}
	_ = map[string]F{"k": id /* ERROR "implicitly instantiated function in map literal requires go1.27 or later" */}
	_ = map[string]P{"k": pair[ /* ERROR "partially instantiated function in map value requires go1.27 or later" */ int]}
	_ = []S{{f: id /* ERROR "implicitly instantiated function in struct literal requires go1.27 or later" */}}

	// literals nested in variable contexts
	var s S = S{f: id /* ERROR "implicitly instantiated function in struct literal requires go1.27 or later" */}
	gs(S{f: id /* ERROR "implicitly instantiated function in struct literal requires go1.27 or later" */})
	var f F = F(id /* ERROR "implicitly instantiated function in conversion requires go1.27 or later" */)
	gv([]F{id /* ERROR "implicitly instantiated function in array or slice literal requires go1.27 or later" */}...)
	var fs []F
	fs = append(fs, []F{id /* ERROR "implicitly instantiated function in array or slice literal requires go1.27 or later" */}...)
	_, _, _ = s, f, fs

	ch := make(chan F, 1)
	ch <- id /* ERROR "implicitly instantiated function in send requires go1.27 or later" */
	chp := make(chan P, 1)
	chp <- pair[ /* ERROR "partially instantiated function in channel send requires go1.27 or later" */ int]
	select {
	case ch <- id /* ERROR "implicitly instantiated function in send requires go1.27 or later" */ :
	}

	_ = F(id /* ERROR "implicitly instantiated function in conversion requires go1.27 or later" */)
	_ = P(pair[ /* ERROR "partially instantiated function in conversion requires go1.27 or later" */ int])
}

// Function type inference when assigning to a variable of type
// parameter type requires Go 1.27 (go.dev/issue/77439).
// Passing to a parameter of type parameter type is permitted
// since Go 1.21.

func _[G ~func(int) int]() G {
	var g G = id /* ERROR "implicitly instantiated function in assignment to type parameter requires go1.27 or later" */
	g = id       /* ERROR "implicitly instantiated function in assignment to type parameter requires go1.27 or later" */
	k := func(G) {}
	k(id)
	k(g)
	return id /* ERROR "implicitly instantiated function in assignment to type parameter requires go1.27 or later" */
}

// No version error for code that is invalid anyway.

func _(f F) {
	switch f {
	case id /* ERROR "cannot use generic function id without instantiation" */ :
	}
}
