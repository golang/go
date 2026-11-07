// -lang=go1.21

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
)

func g(F) {}
func h(P) {}

// At Go 1.21, function type inference is only permitted when assigning
// to a variable (incl. parameters and results).

func _() F {
	var f F = id
	var p P = pair[int]
	f = id
	p = pair[int]
	g(id)
	h(pair[int])
	var fs []F
	fs = append(fs, id)
	fs = append(fs, id, id)
	var ps []P
	ps = append(ps, pair[int])
	_, _, _, _ = f, p, fs, ps

	_ = S{f: id /* ERROR "implicitly instantiated function in struct literal requires go1.27 or later" */}
	_ = []P{pair[ /* ERROR "partially instantiated function in array or slice element requires go1.27 or later" */ int]}
	ch := make(chan F, 1)
	ch <- id /* ERROR "implicitly instantiated function in send requires go1.27 or later" */
	_ = F(id /* ERROR "implicitly instantiated function in conversion requires go1.27 or later" */)
	return id
}
