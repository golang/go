// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Platforms stuck in tsan v2 use a CAS implementation for certain atomic operations.

//go:build race && openbsd && amd64

package atomic

import (
	_ "unsafe" // for go:linkname
)

//go:nosplit
//go:linkname andInt32 sync/atomic.AndInt32
func andInt32(ptr *int32, val int32) int32 {
	for {
		old := *ptr
		if CompareAndSwapInt32(ptr, old, old&val) {
			return old
		}
	}
}

//go:nosplit
//go:linkname orInt32 sync/atomic.OrInt32
func orInt32(ptr *int32, val int32) int32 {
	for {
		old := *ptr
		if CompareAndSwapInt32(ptr, old, old|val) {
			return old
		}
	}
}

//go:nosplit
//go:linkname andInt64 sync/atomic.AndInt64
func andInt64(ptr *int64, val int64) int64 {
	for {
		old := *ptr
		if CompareAndSwapInt64(ptr, old, old&val) {
			return old
		}
	}
}

//go:nosplit
//go:linkname orInt64 sync/atomic.OrInt64
func orInt64(ptr *int64, val int64) int64 {
	for {
		old := *ptr
		if CompareAndSwapInt64(ptr, old, old|val) {
			return old
		}
	}
}
