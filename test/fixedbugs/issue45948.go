// compile -N

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 45948: assert in debug generation for degenerate
// function with infinite loop.

package p

func f(p int) {
L:
	goto L

}
