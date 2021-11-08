// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && !arm64 && !s390x && !ppc64le && !ppc64

package bytealg

const MaxBruteForce = 0

// Index returns the index of the first instance of b in a, or -1 if b is not present in a.
// Requires 2 <= len(b) <= MaxLen.
func Index(a, b []byte) int {
	panic("unimplemented")
}

// IndexString returns the index of the first instance of b in a, or -1 if b is not present in a.
// Requires 2 <= len(b) <= MaxLen.
func IndexString(a, b string) int {
	panic("unimplemented")
}

// Cutover reports the number of failures of IndexByte we should tolerate
// before switching over to Index.
// n is the number of bytes processed so far.
// See the bytes.Index implementation for details.
func Cutover(n int) int {
	panic("unimplemented")
}
