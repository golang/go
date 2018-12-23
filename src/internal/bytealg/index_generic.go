// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64,!arm64,!s390x

package bytealg

const MaxBruteForce = 0

// Index returns the index of the first instance of b in a, or -1 if b is not present in a.
// Requires 2 <= len(b) <= MaxLen.
func Index(a, b []byte) int {
	panic("unimplemented")
}

// IndexString returns the index of the first instance of b in a, or -1 if b is not present in a.
// Requires 2 <= len(b) <= MaxLen.
func IndexString(s, substr string) int {
	// This is a partial copy of strings.Index, here because bytes.IndexAny and bytes.LastIndexAny
	// call bytealg.IndexString. Some platforms have an optimized assembly version of this function.
	// This implementation is used for those that do not. Although the pure Go implementation here
	// works for the case of len(b) > MaxLen, we do not require that its assembly implementation also
	// supports the case of len(b) > MaxLen. And we do not guarantee that this function supports the
	// case of len(b) > MaxLen.
	n := len(substr)
	c0 := substr[0]
	c1 := substr[1]
	i := 0
	t := len(s) - n + 1
	fails := 0
	for i < t {
		if s[i] != c0 {
			o := IndexByteString(s[i:t], c0)
			if o < 0 {
				return -1
			}
			i += o
		}
		if s[i+1] == c1 && s[i:i+n] == substr {
			return i
		}
		i++
		fails++
		if fails >= 4+i>>4 && i < t {
			// See comment in src/bytes/bytes.go.
			j := IndexRabinKarp(s[i:], substr)
			if j < 0 {
				return -1
			}
			return i + j
		}
	}
	return -1
}

// Cutover reports the number of failures of IndexByte we should tolerate
// before switching over to Index.
// n is the number of bytes processed so far.
// See the bytes.Index implementation for details.
func Cutover(n int) int {
	panic("unimplemented")
}
