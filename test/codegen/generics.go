// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "cmp"

func isNaN[T cmp.Ordered](x T) bool {
	return x != x
}

func compare[T cmp.Ordered](x, y T) int {
	// amd64:-"TESTB"
	// arm64:-"MOVB"
	xNaN := isNaN(x)
	yNaN := isNaN(y)
	if xNaN {
		if yNaN {
			return 0
		}
		return -1
	}
	if yNaN {
		return +1
	}
	if x < y {
		return -1
	}
	if x > y {
		return +1
	}
	return 0
}

func usesCompare(a, b int) int {
	return compare(a, b)
}
