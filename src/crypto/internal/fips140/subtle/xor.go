// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package subtle

import "crypto/internal/fips140/alias"

// XORBytes sets dst[i] = x[i] ^ y[i] for all i < n = min(len(x), len(y)),
// returning n, the number of bytes written to dst.
//
// If dst does not have length at least n,
// XORBytes panics without writing anything to dst.
//
// dst and x or y may overlap exactly or not at all,
// otherwise XORBytes may panic.
func XORBytes(dst, x, y []byte) int {
	n := min(len(x), len(y))
	if n == 0 {
		return 0
	}
	if n > len(dst) {
		panic("subtle.XORBytes: dst too short")
	}
	if alias.InexactOverlap(dst[:n], x[:n]) || alias.InexactOverlap(dst[:n], y[:n]) {
		panic("subtle.XORBytes: invalid overlap")
	}
	xorBytes(&dst[0], &x[0], &y[0], n) // arch-specific
	return n
}
