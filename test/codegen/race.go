// asmcheck -race

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// Check that we elide racefuncenter/racefuncexit for
// functions with no calls (but which might panic
// in various ways). See issue 31219.
// amd64:-"CALL.*racefuncenter.*"
func RaceMightPanic(a []int, i, j, k, s int) {
	var b [4]int
	_ = b[i]     // panicIndex
	_ = a[i:j]   // panicSlice
	_ = a[i:j:k] // also panicSlice
	_ = i << s   // panicShift
	_ = i / j    // panicDivide
}
