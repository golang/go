// errorcheck -0 -m -d=retdevirt

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the analysis half of return-value devirtualization: which
// interface-typed results are recorded as always holding a single
// dynamic type.

package p

type I interface{ M() }

type T struct{ x int }

//go:noinline
func (*T) M() {}

type U struct{ x int }

//go:noinline
func (*U) M() {}

//go:noinline
func noop() {}

// A direct return of a single concrete type.

//go:noinline
func newT() I { // ERROR "result #0 of newT is always \*T"
	return &T{} // ERROR "&T{} escapes to heap"
}

// Propagation through a static call to an analyzed function.

//go:noinline
func newTIndirect() I { // ERROR "result #0 of newTIndirect is always \*T"
	return newT()
}

// nil is a dynamic type of its own.

//go:noinline
func newNil() I { // ERROR "result #0 of newNil is always <nil>"
	return nil
}

// Mixing nil with a concrete type records both: callers may compare
// the result against nil, so the concrete type alone would be wrong.

//go:noinline
func newTOrNil(b bool) I { // ERROR "result #0 of newTOrNil is one of <nil>, \*T"
	if b {
		return nil
	}
	return &T{} // ERROR "&T{} escapes to heap"
}

// Two concrete types.

//go:noinline
func newTOrU(b bool) I { // ERROR "result #0 of newTOrU is one of \*U, \*T"
	if b {
		return &U{} // ERROR "&U{} escapes to heap"
	}
	return &T{} // ERROR "&T{} escapes to heap"
}

// A defer may recover, replacing the result with nil.

//go:noinline
func newTDefer() I {
	defer noop()
	return &T{} // ERROR "&T{} escapes to heap"
}

// Bare returns read named results, which are not followed yet.

//go:noinline
func newTNamed() (r I) {
	r = &T{} // ERROR "&T{} escapes to heap"
	return
}

// An interface-typed value of unknown dynamic type.

//go:noinline
func passthrough(x I) I { // ERROR "leaking param: x to result ~r0 level=0"
	return x
}

// Multiple results are analyzed independently.

//go:noinline
func pair() (I, error) { // ERROR "result #0 of pair is always \*T" "result #1 of pair is always <nil>"
	return &T{}, nil // ERROR "&T{} escapes to heap"
}

// Multi-valued propagation through "return g()".

//go:noinline
func pairIndirect() (I, error) { // ERROR "result #0 of pairIndirect is always \*T" "result #1 of pairIndirect is always <nil>"
	return pair()
}

// A concrete result of the callee converted to an interface result of
// the caller by a multi-valued return.

//go:noinline
func concretePair() (*T, error) { // ERROR "result #1 of concretePair is always <nil>"
	return &T{}, nil // ERROR "&T{} escapes to heap"
}

//go:noinline
func pairFromConcrete() (I, error) { // ERROR "result #0 of pairFromConcrete is always \*T" "result #1 of pairFromConcrete is always <nil>"
	return concretePair()
}

// A local with a single assignment is followed to its value.

//go:noinline
func newTLocal() I { // ERROR "result #0 of newTLocal is always \*T"
	var v I = &T{} // ERROR "&T{} escapes to heap"
	return v
}

// A declaration without an initializer contributes nil: the zero
// value of an interface variable is observable when no assignment
// runs. This is load-bearing for correctness; see the split test.

//go:noinline
func newTOrZero(b bool) I { // ERROR "result #0 of newTOrZero is one of <nil>, \*T"
	var v I
	if b {
		v = &T{} // ERROR "&T{} escapes to heap"
	}
	return v
}

// A reassigned local contributes every assignment to the set.

//go:noinline
func newTOrULocal(b bool) I { // ERROR "result #0 of newTOrULocal is one of \*T, \*U"
	var v I = &T{} // ERROR "&T{} escapes to heap"
	if b {
		v = &U{} // ERROR "&U{} escapes to heap"
	}
	return v
}

// The recorded result sets feed receiver devirtualization: a method
// call on the result of a static call to an analyzed function becomes
// a direct call, with no inlining involved.

//go:noinline
func callM() {
	v := newT()
	v.M() // ERROR "devirtualizing v.M to \*T"
}

// Recursive functions see no recorded types for their own component.

//go:noinline
func recursive(n int) I {
	if n == 0 {
		return &T{} // ERROR "&T{} escapes to heap"
	}
	return recursive(n - 1)
}

// A shaped function whose result does not depend on the dictionary is
// recorded like any other function, and its callers devirtualize.

//go:noinline
func newTShaped[X any]() I { // ERROR "result #0 of newTShaped\[go\.shape\.int\] is always \*T" "result #0 of newTShaped\[int\] is always \*T"
	return &T{} // ERROR "&T{} escapes to heap"
}

// A result whose dynamic type depends on the dictionary is unknown.

type box[X any] struct{ x X }

//go:noinline
func (*box[X]) M() {}

//go:noinline
func newBox[X any]() I {
	return &box[X]{} // ERROR "escapes to heap"
}

//go:noinline
func callShaped() {
	v := newTShaped[int]()
	v.M() // ERROR "devirtualizing v.M to \*T"
	w := newBox[int]()
	w.M()
}
