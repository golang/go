// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that the shortcircuit pass correctly handles infinite loops.

package p

func f() {
	var p, q bool
	for {
		p = p && q
	}
}
