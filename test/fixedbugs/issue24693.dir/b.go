// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

type T struct{ a.T }

func (T) m() { println("ok") }

// The compiler used to not pay attention to package for non-exported
// methods when statically constructing itabs. The consequence of this
// was that the call to b.F1(b.T{}) in c.go would create an itab using
// a.T.m instead of b.T.m.
func F1(i interface{ m() }) { i.m() }

// The interface method calling convention depends on interface method
// sets being sorted in the same order across compilation units.  In
// the test case below, at the call to b.F2(b.T{}) in c.go, the
// interface method set is sorted as { a.m(); b.m() }.
//
// However, while compiling package b, its package path is set to "",
// so the code produced for F2 uses { b.m(); a.m() } as the method set
// order. So again, it ends up calling the wrong method.
//
// Also, this function is marked noinline because it's critical to the
// test that the interface method call happen in this compilation
// unit, and the itab construction happens in c.go.
//
//go:noinline
func F2(i interface {
	m()
	a.I // embeds m() from package a
}) {
	i.m()
}
