// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm

package bigmod

func makeWideLimbs(n int) []uint {
	// Attempt to use a stack-allocated backing array.
	T := make([]uint, 0, preallocLimbs*2)
	if cap(T) < n*2 {
		T = make([]uint, 0, n*2)
	}
	return T[:n*2]
}
