// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program results in a loop inferred to increment
// j by 0, causing bounds check elimination to attempt
// something%0, which panics (in the bug).

package q

func f() {
	var s1 string
	var b bool
	if b {
		b = !b
		s1 += "a"
	}

	var s2 string
	var i, j int
	if (s1 <= "") || (s2 >= "") {
		j = len(s1[:6])
	} else {
		i = len("b")
	}

	for j < 0 {
		j += i
	}
}
