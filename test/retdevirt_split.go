// errorcheck -0 -m -d=retdevirt

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the transform half of return-value devirtualization: which
// functions split into a devirtualized variant plus a thunk, and how
// static calls to them are rewritten.

package p

type I interface{ M() int }

type T struct{ x int }

//go:noinline
func (t *T) M() int { return t.x } // ERROR "t does not escape"

//go:noinline
func grow() {}

// Non-inlinable and always one concrete type: splits.

func newT(n int) I { // ERROR "result #0 of newT is always \*T" "splitting newT into newT\.dv"
	if n < 0 {
		grow()
		grow()
	}
	return &T{x: n} // ERROR "&T{...} escapes to heap"
}

// An argless variant, for the defer test below.

var neg bool

func newT0() I { // ERROR "result #0 of newT0 is always \*T" "splitting newT0 into newT0\.dv"
	if neg {
		grow()
		grow()
	}
	return &T{} // ERROR "&T{} escapes to heap"
}

// The rewrite exposes the concrete type, so the method call on the
// result devirtualizes with no inlining involved.

//go:noinline
func use() int {
	v := newT(1) // ERROR "devirtualizing call to newT\.dv"
	return v.M() // ERROR "devirtualizing v.M to \*T"
}

// Only the qualifying result slot changes type; the always-nil error
// slot keeps its interface type.

func newTPair(n int) (I, error) { // ERROR "result #0 of newTPair is always \*T" "result #1 of newTPair is always <nil>" "splitting newTPair into newTPair\.dv"
	if n < 0 {
		grow()
		grow()
	}
	return &T{x: n}, nil // ERROR "&T{...} escapes to heap"
}

//go:noinline
func usePair() int {
	v, err := newTPair(1) // ERROR "devirtualizing call to newTPair\.dv"
	if err != nil {
		return 0
	}
	return v.M() // ERROR "devirtualizing v.M to \*T"
}

// Small and inlinable: left to the inliner, which exposes the
// concrete type by inlining the body instead.

func newTSmall() I { // ERROR "can inline newTSmall" "result #0 of newTSmall is always \*T"
	return &T{} // ERROR "&T{} escapes to heap"
}

//go:noinline
func useSmall() int {
	v := newTSmall() // ERROR "inlining call to newTSmall" "&T{} does not escape"
	return v.M()     // ERROR "devirtualizing v.M to \*T"
}

// A pragma disables splitting; the result set is still recorded.

//go:noinline
func newTNoinline(n int) I { // ERROR "result #0 of newTNoinline is always \*T"
	if n < 0 {
		grow()
		grow()
	}
	return &T{x: n} // ERROR "&T{...} escapes to heap"
}

// A deferred call to a split function is normalized into a
// synthesized closure first (its results alone force that), and the
// call inside the closure is rewritten like any other; the closure
// still runs at return. StaticResults refuses the rewrite for the
// plain deferred calls that skip normalization.

//go:noinline
func useDefer() {
	defer newT0() // ERROR "can inline useDefer\.deferwrap1" "devirtualizing call to newT0\.dv"
}

// Methods split too: the receiver is promoted to a leading parameter
// of the variant, and the method itself stays behind as the thunk
// that itabs, method values, and reflection use.

type maker struct{ n int }

func (m *maker) Make() I { // ERROR "m does not escape" "result #0 of \(\*maker\)\.Make is always \*T" "splitting \(\*maker\)\.Make into \(\*maker\)\.Make\.dv"
	if m.n < 0 {
		grow()
		grow()
	}
	return &T{x: m.n} // ERROR "&T{...} escapes to heap"
}

//go:noinline
func useMethod(m *maker) int { // ERROR "m does not escape"
	v := m.Make() // ERROR "devirtualizing call to \(\*maker\)\.Make\.dv"
	return v.M()  // ERROR "devirtualizing v.M to \*T"
}

// Devirtualizations chain: the interface receiver devirtualizes,
// which makes the method call static, which rewrites it to the
// variant, which devirtualizes the method call on its result.

type mk interface{ Make() I }

//go:noinline
func useChain(m *maker) int { // ERROR "m does not escape"
	var i mk = m
	v := i.Make() // ERROR "devirtualizing i.Make to \*maker" "devirtualizing call to \(\*maker\)\.Make\.dv"
	return v.M()  // ERROR "devirtualizing v.M to \*T"
}

// Shaped functions split per shape; the dictionary is forwarded like
// any parameter, and the sets are dictionary-independent by
// construction. The extra messages come from the instantiation
// wrapper, whose body's call is rewritten like any other.

func newTStencil[X any](n int) I { // ERROR "result #0 of newTStencil\[go\.shape\.int\] is always \*T" "splitting newTStencil\[go\.shape\.int\] into newTStencil\[go\.shape\.int\]\.dv" "result #0 of newTStencil\[int\] is always \*T" "can inline newTStencil\[int\]" "devirtualizing call to newTStencil\[go\.shape\.int\]\.dv"
	if n < 0 {
		grow()
		grow()
	}
	return &T{x: n} // ERROR "&T{...} escapes to heap"
}

//go:noinline
func useStencil() int {
	v := newTStencil[int](1) // ERROR "devirtualizing call to newTStencil\[go\.shape\.int\]\.dv"
	return v.M()             // ERROR "devirtualizing v.M to \*T"
}

// A body with function literals splits too: captures of moved locals
// keep their identity, and captures of substituted parameters are
// rebound.

func newTClosure(n int) I { // ERROR "result #0 of newTClosure is always \*T" "splitting newTClosure into newTClosure\.dv"
	double := n * 2
	cb := func() int { return n + double } // ERROR "can inline newTClosure\.func1"
	if n < 0 {
		grow()
		grow()
	}
	return &T{x: cb()} // ERROR "&T{...} escapes to heap" "inlining call to newTClosure\.func1"
}

//go:noinline
func useClosure() int {
	v := newTClosure(3) // ERROR "devirtualizing call to newTClosure\.dv"
	return v.M()        // ERROR "devirtualizing v.M to \*T"
}

// A named result captured by a literal cannot change type; its slot
// is dropped and, it being the only one, the function is not split.

func newTCaptured(n int) (r I) { // ERROR "result #0 of newTCaptured is always \*T"
	probe := func() I { return r } // ERROR "can inline newTCaptured\.func1" "func literal does not escape"
	_ = probe
	if n < 0 {
		grow()
		grow()
	}
	return &T{x: n} // ERROR "&T{...} escapes to heap"
}

// A mid-cost function is inlinable and also splits. At call sites
// the inliner reaches first, so the original inlines as before; the
// variant serves the sites inlining declines, and can itself inline
// through the original's body should such a site's conditions later
// allow it.

func newTMid(n int) I { // ERROR "can inline newTMid" "result #0 of newTMid is always \*T" "splitting newTMid into newTMid\.dv"
	if n < 0 {
		grow()
	}
	return &T{x: n} // ERROR "&T{...} escapes to heap"
}

//go:noinline
func useMid() int {
	v := newTMid(1) // ERROR "inlining call to newTMid" "&T{...} does not escape"
	return v.M()    // ERROR "devirtualizing v.M to \*T"
}
