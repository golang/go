// errorcheck -0 -live -l -d=compilelater,eagerwb

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 20250: liveness differed with concurrent compilation
// due to propagation of addrtaken to outer variables for
// closure variables.

// TODO(austin): This expects function calls to the write barrier, so
// we enable the legacy eager write barrier. Fix this once the
// buffered write barrier works on all arches.

package p

type T struct {
	s string
}

func f(a T) { // ERROR "live at entry to f: a"
	var e interface{}
	func() { // ERROR "live at entry to f.func1: a &e"
		e = a.s // ERROR "live at call to convT2Estring: a &e" "live at call to writebarrierptr: a"
	}() // ERROR "live at call to f.func1: e$"
	// Before the fix, both a and e were live at the previous line.
	_ = e
}
