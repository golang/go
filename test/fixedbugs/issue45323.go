// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func g() bool

func f(y int) bool {
	b, ok := true, false
	if y > 1 {
		ok = g()
	}
	if !ok {
		ok = g()
		b = false
	}
	if !ok {
		return false
	}
	return b
}
