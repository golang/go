// compile

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test legal shifts.
// Issue 1708, legal cases.
// Compiles but does not run.

package p

func f(x int) int         { return 0 }
func g(x interface{}) int { return 0 }
func h(x float64) int     { return 0 }

// from the spec
var (
	s uint  = 33
	i       = 1 << s         // 1 has type int
	j int32 = 1 << s         // 1 has type int32; j == 0
	k       = uint64(1 << s) // 1 has type uint64; k == 1<<33
	l       = g(1 << s)      // 1 has type int
	m int   = 1.0 << s       // legal: 1.0 has type int
	w int64 = 1.0 << 33      // legal: 1.0<<33 is a constant shift expression
)

// non-constant shift expressions
var (
	a1 int = 2.0 << s    // typeof(2.0) is int in this context => legal shift
	d1     = f(2.0 << s) // typeof(2.0) is int in this context => legal shift
)

// constant shift expressions
const c uint = 5

var (
	a2 int     = 2.0 << c    // a2 == 64 (type int)
	b2         = 2.0 << c    // b2 == 64 (untyped integer)
	_          = f(b2)       // verify b2 has type int
	c2 float64 = 2 << c      // c2 == 64.0 (type float64)
	d2         = f(2.0 << c) // == f(64)
	e2         = g(2.0 << c) // == g(int(64))
	f2         = h(2 << c)   // == h(float64(64.0))
)
