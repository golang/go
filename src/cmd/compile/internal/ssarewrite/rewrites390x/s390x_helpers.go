// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rewrites390x

// is8Bit reports whether n can be represented as a signed 8 bit integer.
func is8Bit(n int64) bool {
	return n == int64(int8(n))
}

// isU12Bit reports whether n can be represented as an unsigned 12 bit integer.
func isU12Bit(n int64) bool {
	return 0 <= n && n < (1<<12)
}

// isU8Bit reports whether n can be represented as an unsigned 8 bit integer.
func isU8Bit(n int64) bool {
	return n == int64(uint8(n))
}
