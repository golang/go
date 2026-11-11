// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytealg

// bytes.Index and strings.Index pass haystacks no longer than this
// straight to Index, rather than first skipping ahead with IndexByte.
const MaxBruteForce = 64

func init() {
	// Optimize cases where the length of the substring is less than 32 bytes
	MaxLen = 32
}

// Cutover reports the number of failures of IndexByte we should tolerate
// before switching over to Index.
// n is the number of bytes processed so far.
// See the bytes.Index implementation for details.
func Cutover(n int) int {
	// 1 error per 16 characters, plus a few slop to start.
	return 4 + n>>4
}
