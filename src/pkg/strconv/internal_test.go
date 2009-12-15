// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// export access to strconv internals for tests

package strconv

func NewDecimal(i uint64) *decimal { return newDecimal(i) }

func SetOptimize(b bool) bool {
	old := optimize
	optimize = b
	return old
}
