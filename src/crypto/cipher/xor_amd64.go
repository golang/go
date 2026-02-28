// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher

// xorBytes xors the bytes in a and b. The destination should have enough
// space, otherwise xorBytes will panic. Returns the number of bytes xor'd.
func xorBytes(dst, a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	_ = dst[n-1]
	xorBytesSSE2(&dst[0], &a[0], &b[0], n) // amd64 must have SSE2
	return n
}

func xorWords(dst, a, b []byte) {
	xorBytes(dst, a, b)
}

//go:noescape
func xorBytesSSE2(dst, a, b *byte, n int)
