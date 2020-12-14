// errorcheck

// +build amd64

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 20529: Large stack frames caused compiler panics.
// Only tested on amd64 because the test only makes sense
// on a 64 bit system, and it is platform-agnostic,
// so testing one suffices.

package p

import "runtime"

func f() { // GC_ERROR "stack frame too large"
	x := [][]int{1e9: []int{}}
	runtime.KeepAlive(x)
}
