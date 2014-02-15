// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that code compiles without
// "internal error: ... recorded as live on entry" errors
// from the liveness code.
//
// This code contains methods or other construct that
// trigger the generation of wrapper functions with no
// clear line number (they end up using line 1), and those
// would have annotations printed if we used -live=1,
// like the live.go test does.
// Instead, this test relies on the fact that the liveness
// analysis turns any non-live parameter on entry into
// a compile error. Compiling successfully means that bug
// has been avoided.

package main

// The liveness analysis used to get confused by the tail return
// instruction in the wrapper methods generated for T1.M and (*T1).M,
// causing a spurious "live at entry: ~r1" for the return result.

type T struct {
}

func (t *T) M() *int

type T1 struct {
	*T
}

// Liveness analysis used to have the VARDEFs in the wrong place,
// causing a temporary to appear live on entry.

func f1(pkg, typ, meth string) {
	panic("value method " + pkg + "." + typ + "." + meth + " called using nil *" + typ + " pointer")
}

func f2() interface{} {
	return new(int)
}

