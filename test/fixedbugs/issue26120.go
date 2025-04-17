// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 26120: INDEX of 1-element but non-SSAable array
// is mishandled when building SSA.

package p

type T [1]struct {
	f    []int
	i, j int
}

func F() {
	var v T
	f := func() T {
		return v
	}
	_ = []int{}[f()[0].i]
}
