// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check for cycles in the method value of a value returned from a function call.

package funcmethvalue

type T int

func (T) m() int {
	_ = x
	return 0
}

func f() T {
	return T(0)
}

var (
	t T
	x = f().m // ERROR "initialization loop|depends upon itself"
)
