// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// The rotate amounts are only rewritten on 32-bit platforms,
// so this only failed to compile there.

func rotate(x uint32, n int) uint32 {
	if n < 0 {
		s := uint(-n)
		return x<<s | x>>(32-s)
	}
	s := uint(n)
	return x>>s | x<<(32-s)
}

func f(x uint32) uint32 {
	return rotate(rotate(x, 7), -7)
}
