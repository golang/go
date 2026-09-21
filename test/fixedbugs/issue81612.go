// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The stack-allocated slice backing store optimization must not kick in
// when the address of a field of a slice element escapes. &s[i].f points
// into s's backing store just like &s[i] does, so the backing store has
// to be moved to the heap.

package main

type def struct {
	ID int64
}

//go:noinline
func resolve() *int64 {
	defs := []def{{77}}
	var matches []def
	for _, d := range defs {
		matches = append(matches, d)
	}
	switch len(matches) {
	case 1:
		return &matches[0].ID
	default:
		candidates := make([]int64, 0, len(matches))
		for _, m := range matches {
			candidates = append(candidates, m.ID)
		}
		return &candidates[0]
	}
}

var sink int64
var escaped []def

// resolveNoRange reaches the same bug without ranging over the slice:
// here the exclusive->nonexclusive transition is the assignment to
// escaped, which is on a different path than the &matches[0].ID.
//
//go:noinline
func resolveNoRange(n int) *int64 {
	var matches []def
	for i := range n {
		matches = append(matches, def{int64(77 + i)})
	}
	if len(matches) == 1 {
		return &matches[0].ID
	}
	escaped = matches
	return &sink
}

type elem struct{ a [4]int64 }

var escapedElems []elem

// resolveViaSliceArr reaches the same bug through a slice of an array
// field: &t[0].a[:][0] points into t's backing store just as &t[0].a[0]
// does, because the OSLICEARR shares storage with the array.
//
//go:noinline
func resolveViaSliceArr(n int) *int64 {
	var t []elem
	for i := range n {
		t = append(t, elem{a: [4]int64{int64(77 + i)}})
	}
	if len(t) == 1 {
		return &t[0].a[:][0]
	}
	escapedElems = t
	return &sink
}

// clobber overwrites the stack frames below main's.
//
//go:noinline
func clobber(n int) {
	var buf [256]int64
	for i := range buf {
		buf[i] = 0xdeadbeef
	}
	if n > 0 {
		clobber(n - 1)
	}
	sink = buf[n&255]
}

func main() {
	p := resolve()
	if *p != 77 {
		println("resolve: before clobber:", *p)
		panic("wrong value")
	}
	clobber(4)
	if *p != 77 {
		println("resolve: after clobber:", *p)
		panic("value destroyed by unrelated stack traffic")
	}

	p = resolveNoRange(1)
	if *p != 77 {
		println("resolveNoRange: before clobber:", *p)
		panic("wrong value")
	}
	clobber(4)
	if *p != 77 {
		println("resolveNoRange: after clobber:", *p)
		panic("value destroyed by unrelated stack traffic")
	}

	p = resolveViaSliceArr(1)
	if *p != 77 {
		println("resolveViaSliceArr: before clobber:", *p)
		panic("wrong value")
	}
	clobber(4)
	if *p != 77 {
		println("resolveViaSliceArr: after clobber:", *p)
		panic("value destroyed by unrelated stack traffic")
	}
}
