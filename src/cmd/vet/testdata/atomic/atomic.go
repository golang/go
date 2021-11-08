// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the atomic checker.

package atomic

import "sync/atomic"

func AtomicTests() {
	x := uint64(1)
	x = atomic.AddUint64(&x, 1) // ERROR "direct assignment to atomic value"
}
