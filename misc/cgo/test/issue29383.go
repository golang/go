// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// cgo's /*line*/ comments failed when inserted after '/',
// because the result looked like a "//" comment.
// No runtime test; just make sure it compiles.

package cgotest

// #include <stddef.h>
import "C"

func Issue29383(n, size uint) int {
	if ^C.size_t(0)/C.size_t(n) < C.size_t(size) {
		return 0
	}
	return 0
}
