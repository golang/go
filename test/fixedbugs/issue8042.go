// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that gotos across non-variable declarations
// are accepted.

package p

func _() {
	goto L1
	const x = 0
L1:
	goto L2
	type T int
L2:
}

func _() {
	{
		goto L1
	}
	const x = 0
L1:
	{
		goto L2
	}
	type T int
L2:
}

func _(d int) {
	if d > 0 {
		goto L1
	} else {
		goto L2
	}
	const x = 0
L1:
	switch d {
	case 1:
		goto L3
	case 2:
	default:
		goto L4
	}
	type T1 int
L2:
	const y = 1
L3:
	for d > 0 {
		if d < 10 {
			goto L4
		}
	}
	type T2 int
L4:
	select {
	default:
		goto L5
	}
	type T3 int
L5:
}
