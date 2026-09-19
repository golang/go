// run -gcflags=-d=retdevirt

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check the runtime semantics of split functions: boxing, nil
// comparisons, type assertions, indirect calls through the thunk,
// and defer timing through the normalized closure.

package main

import (
	"fmt"
	"reflect"
)

type I interface{ M() int }

type T struct{ x int }

//go:noinline
func (t *T) M() int { return t.x }

//go:noinline
func grow() {}

var order []int

// Non-inlinable via two calls; splits under -d=retdevirt.
func newT(n int) I {
	order = append(order, n)
	if n < -1000 {
		grow()
		grow()
	}
	return &T{x: n}
}

func newTPair(n int) (I, error) {
	if n < -1000 {
		grow()
		grow()
	}
	return &T{x: n}, nil
}

type maker struct{ n int }

func (m *maker) Make() I {
	if m.n < -1000 {
		grow()
		grow()
	}
	return &T{x: m.n}
}

func (m maker) MakeVal() I {
	if m.n < -1000 {
		grow()
		grow()
	}
	return &T{x: m.n + 1}
}

type mk interface{ Make() I }

func deferCheck() {
	defer newT(10)
	order = append(order, 1)
}

// Shaped constructor: one body per shape, dictionary forwarded.
func newG[X any](n int) I {
	if n < -1000 {
		grow()
		grow()
	}
	return &T{x: n}
}

// Closure-bearing constructor: captures a parameter and a local.
func newCB(n int) I {
	double := n * 2
	cb := func() int { return n + double }
	if n < -1000 {
		grow()
		grow()
	}
	return &T{x: cb()}
}

// The named result is captured by the literal; its slot must keep
// its interface type, and the capture must observe the final value.
func newCap(n int) (I, func() I) {
	var probe func() I
	f := func() (r I) {
		probe = func() I { return r }
		if n < -1000 {
			grow()
			grow()
		}
		r = &T{x: n}
		return r
	}
	v := f()
	return v, probe
}

// A value type boxed indirectly: unboxing it loads through the data
// pointer rather than taking the word itself.
type big struct{ a, b, c int }

func (v big) M() int { return v.a + v.b + v.c }

// Recorded {big} but not split, so the split callers of mkBig unbox
// a value whose boxing site they cannot see.
//
//go:noinline
func mkBig(n int) I {
	if n < -1000 {
		grow()
		grow()
	}
	return big{a: n, b: n, c: n}
}

func viaBig(n int) I {
	if n < -1000 {
		grow()
		grow()
	}
	return mkBig(n)
}

// Mid-cost: inlinable and split, so calls inline the variant. The
// value-type result exercises the inlined deref unboxing.
func midBig(n int) I {
	if n < -1000 {
		grow()
	}
	return big{a: n, b: n, c: n}
}

func midT(n int) I {
	if n < -1000 {
		grow()
	}
	return &T{x: n}
}

type errno int

func (errno) Error() string { return "errno" }

// The internal/syscall/unix.Getaddrinfo shape: an interface variable
// declared at its zero value and only conditionally assigned. The
// nil member recorded for the declaration must prevent the split.
func maybeErr(b bool) (int, error) {
	if b {
		grow()
		grow()
	}
	var err error
	if b {
		err = errno(7)
	}
	return 42, err
}

func main() {
	check := func(cond bool, msg string) {
		if !cond {
			panic(msg)
		}
	}

	v := newT(7)
	check(v != nil, "v == nil")
	check(v.M() == 7, "v.M() != 7")

	w, err := newTPair(3)
	check(err == nil, "err != nil")
	check(w.M() == 3, "w.M() != 3")

	_, ok := v.(*T)
	check(ok, "v.(*T) failed")
	check(fmt.Sprintf("%T", v) == "*main.T", "wrong dynamic type")

	// Indirect calls go through the boxing thunk.
	fp := newT
	check(fp(9).M() == 9, "fp(9).M() != 9")

	// So does reflection.
	rv := reflect.ValueOf(newT).Call([]reflect.Value{reflect.ValueOf(11)})
	check(rv[0].Interface().(I).M() == 11, "reflect call failed")

	// A deferred call with arguments runs at return, not at the
	// defer statement.
	order = nil
	deferCheck()
	check(len(order) == 2 && order[0] == 1 && order[1] == 10, "defer ran at the wrong time")

	// Split methods: direct call, method value, interface dispatch
	// through the thunk, and reflection through the thunk.
	m := &maker{n: 5}
	check(m.Make().M() == 5, "method call")
	check(maker{n: 5}.MakeVal().M() == 6, "value receiver method call")
	mv := m.Make
	check(mv().M() == 5, "method value")
	var mi mk = m
	check(mi.Make().M() == 5, "interface dispatch")
	rm := reflect.ValueOf(m).MethodByName("Make").Call(nil)
	check(rm[0].Interface().(I).M() == 5, "reflect method call")

	// Shaped and closure-bearing constructors behave identically
	// when split.
	check(newG[int](21).M() == 21, "shaped int")
	check(newG[string](22).M() == 22, "shaped string")
	check(newCB(5).M() == 15, "closure capture")
	v2, probe := newCap(3)
	check(v2.M() == 3, "captured result value")
	check(probe().M() == 3, "capture observes the result")

	// Inlined variants: pointer and indirectly boxed value results.
	check(midT(31).M() == 31, "inlined variant, pointer result")
	check(midBig(4).M() == 12, "inlined variant, value result")
	mv2 := midBig(2)
	if b3, ok := mv2.(big); !ok || b3.M() != 6 {
		panic("inlined variant identity")
	}

	// Indirectly boxed value types survive the unchecked unboxing.
	check(viaBig(4).M() == 12, "indirect unbox")
	bv := viaBig(2)
	if b2, ok := bv.(big); !ok || b2.M() != 6 {
		panic("indirect unbox identity")
	}

	// The zero value of a conditionally assigned error result stays
	// a nil interface.
	n, err2 := maybeErr(false)
	check(n == 42 && err2 == nil, "zero-value error result is not nil")
	_, err3 := maybeErr(true)
	check(err3 != nil, "assigned error result is nil")
}
