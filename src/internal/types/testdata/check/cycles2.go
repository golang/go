// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

// Test case for issue 5090

type t interface {
	f(u)
}

type u interface {
	t
}

func _() {
	var t t
	var u u

	t.f(t)
	t.f(u)

	u.f(t)
	u.f(u)
}


// Test case for issues #6589, #33656.

type A interface {
	a() interface {
		AB
	}
}

type B interface {
	b() interface {
		AB
	}
}

type AB interface {
	a() interface {
		A
		B
	}
	b() interface {
		A
		B
	}
}

var x AB
var y interface {
	A
	B
}

var _ = x == y


// Test case for issue 6638.

type T interface {
	m() [T(nil).m /* ERROR "undefined" */ ()[0]]int
}

// Variations of this test case.

type T1 /* ERROR "invalid recursive type" */ interface {
	m() [x1.m()[0]]int
}

var x1 T1

type T2 /* ERROR "invalid recursive type" */ interface {
	m() [len(x2.m())]int
}

var x2 T2

type T3 /* ERROR "invalid recursive type" */ interface {
	m() [unsafe.Sizeof(x3.m)]int
}

var x3 T3

type T4 /* ERROR "invalid recursive type" */ interface {
	m() [unsafe.Sizeof(cast4(x4.m))]int // cast is invalid but we have a cycle, so all bets are off
}

var x4 T4
var _ = cast4(x4.m)

type cast4 func()
