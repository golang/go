// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// export access to strconv internals for tests

package strconv

import "runtime"

func NewDecimal(i uint64) *decimal { return newDecimal(i) }

func SetOptimize(b bool) bool {
	if runtime.GOARCH == "arm" {
		// optimize is always false on arm,
		// because the software floating point
		// has such terrible multiplication.
		return false
	}
	old := optimize
	optimize = b
	return old
}
