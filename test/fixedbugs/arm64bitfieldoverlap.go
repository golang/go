// compile -N

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// On arm64 the SBFX -> SBFIZ rewrite produced a negative bitfield
// width when the left shift moved the extracted field entirely out
// of range, generating an ICE.

package p

func f(x int64) int64 {
	return int64(int8(x<<16)) >> 1
}
