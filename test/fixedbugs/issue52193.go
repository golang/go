// errorcheck -0 -m

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Test that inlining doesn't break if devirtualization exposes a new
// inlinable callee.

func f() { // ERROR "can inline f"
	var i interface{ m() } = T(0) // ERROR "T\(0\) does not escape"
	i.m()                         // ERROR "devirtualizing i.m" "inlining call to T.m"
}

type T int

func (T) m() { // ERROR "can inline T.m"
	if never {
		f() // ERROR "inlining call to f" "devirtualizing i.m" "T\(0\) does not escape"
	}
}

var never bool
