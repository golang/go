// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const (
	// Constants that we use and will transfer to the runtime.
	minHeapAlign = 8
	maxSmallSize = 32 << 10
	smallSizeDiv = 8
	smallSizeMax = 1024
	largeSizeDiv = 128
	pageShift    = 13
	tinySize     = 16

	// Derived constants.
	pageSize = 1 << pageShift
)

const (
	maxPtrSize = max(4, 8)
	maxPtrBits = 8 * maxPtrSize

	// Maximum size to generate size specialized functions for.
	// We've seen very limited benefit for specialized functions for larger
	// size classes, and with the wrapper they are sometimes slower
	// than the non-specialized functions.
	// This must match the constant in the compiler.
	specializedMallocMax = 80
)
