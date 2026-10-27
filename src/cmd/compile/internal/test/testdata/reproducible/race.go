// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Constructs that reach -race-specific code in the concurrent backend.

package p

type S struct {
	a, b int
	s    string
	arr  [4]int64
}

var (
	g  S
	gs []S
)

// Multi-field structs are not SSA-able under -race.
func Copy(dst, src *S) { *dst = *src }

func CopyArrays(xs, ys [][8]int) {
	for i := range xs {
		xs[i] = ys[i]
	}
}

func Equal(x, y S) bool { return x == y }

func Select(c1, c2 chan S) S {
	select {
	case v := <-c1:
		return v
	case c2 <- g:
	}
	return S{}
}

func Defer(p *S) (r int) {
	defer func() { r += p.a }()
	return p.b
}

func Closure(xs []S) func() int {
	n := 0
	return func() int {
		for _, x := range xs {
			n += x.a
		}
		return n
	}
}

func Swap[T any](p *T, v T) T {
	old := *p
	*p = v
	return old
}

func UseSwap() {
	var a S
	Swap(&a, g)
	var b [4]string
	Swap(&b, b)
}

//go:norace
func NoRace(p *S) { p.a++ }

func CallNoRace(p *S) {
	NoRace(p)
	p.b++
}

func Map(m map[string]S, k string) S {
	v := m[k]
	m[k] = g
	return v
}

func Append(s S) { gs = append(gs, s) }
