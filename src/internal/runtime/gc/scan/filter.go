// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scan

import "unsafe"

// FilterNil packs non-nil (non-zero) values in bufp together
// at the beginning of bufp, returning the length of the
// packed buffer. It treats bufp as an array of size n.
func FilterNil(bufp *uintptr, n int32) int32 {
	buf := unsafe.Slice(bufp, int(n))
	lo := 0
	hi := len(buf) - 1
	for lo < hi {
		for lo < hi && buf[hi] == 0 {
			hi--
		}
		for lo < hi && buf[lo] != 0 {
			lo++
		}
		if lo >= hi {
			break
		}
		buf[lo] = buf[hi]
		hi--
	}
	if hi >= 0 && buf[hi] == 0 {
		hi--
	}
	return int32(hi) + 1
}
