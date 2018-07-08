// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure dead write barriers are handled correctly.

package main

func f(p **int) {
	// The trick here is to eliminate the block containing the write barrier,
	// but only after the write barrier branches are inserted.
	// This requires some delicate code.
	i := 0
	var b []bool
	var s string
	for true {
		if b[i] {
			var a []string
			s = a[len(s)]
			i = 0
		}
		*p = nil
	}
}
