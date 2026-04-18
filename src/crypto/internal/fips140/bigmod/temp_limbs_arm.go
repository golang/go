// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm

package bigmod

// Returning the slice from a non-inlined helper forces the backing array to
// outlive this stack frame, which avoids arm compiler/runtime miscompilations
// seen when the generic Montgomery temporaries are stack allocated.
//
//go:noinline
func makeWideLimbs(n int) []uint {
	return make([]uint, n*2)
}
