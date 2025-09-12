// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytealg

import "internal/cpu"

// Empirical data shows that using Index can get better
// performance when len(s) <= 16.
const MaxBruteForce = 16

func init() {
	// If SIMD is supported, optimize the cases where the substring length is less than 64 bytes,
	// otherwise, cases the length less than 32 bytes is optimized.
	if cpu.Loong64.HasLASX || cpu.Loong64.HasLSX {
		MaxLen = 64
	} else {
		MaxLen = 32
	}
}

// Cutover reports the number of failures of IndexByte we should tolerate
// before switching over to Index.
// n is the number of bytes processed so far.
// See the bytes.Index implementation for details.
func Cutover(n int) int {
	// 1 error per 8 characters, plus a few slop to start.
	return (n + 16) / 8
}
