// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lcs

// This file defines the abstract sequence over which the LCS algorithm operates.

// sequences abstracts a pair of sequences, A and B.
type sequences interface {
	lengths() (int, int)                    // len(A), len(B)
	commonPrefixLen(ai, aj, bi, bj int) int // len(commonPrefix(A[ai:aj], B[bi:bj]))
	commonSuffixLen(ai, aj, bi, bj int) int // len(commonSuffix(A[ai:aj], B[bi:bj]))
}

type stringSeqs struct{ a, b string }

func (s stringSeqs) lengths() (int, int) { return len(s.a), len(s.b) }
func (s stringSeqs) commonPrefixLen(ai, aj, bi, bj int) int {
	return commonPrefixLenString(s.a[ai:aj], s.b[bi:bj])
}
func (s stringSeqs) commonSuffixLen(ai, aj, bi, bj int) int {
	return commonSuffixLenString(s.a[ai:aj], s.b[bi:bj])
}

// The explicit capacity in s[i:j:j] leads to more efficient code.

type bytesSeqs struct{ a, b []byte }

func (s bytesSeqs) lengths() (int, int) { return len(s.a), len(s.b) }
func (s bytesSeqs) commonPrefixLen(ai, aj, bi, bj int) int {
	return commonPrefixLenBytes(s.a[ai:aj:aj], s.b[bi:bj:bj])
}
func (s bytesSeqs) commonSuffixLen(ai, aj, bi, bj int) int {
	return commonSuffixLenBytes(s.a[ai:aj:aj], s.b[bi:bj:bj])
}

type runesSeqs struct{ a, b []rune }

func (s runesSeqs) lengths() (int, int) { return len(s.a), len(s.b) }
func (s runesSeqs) commonPrefixLen(ai, aj, bi, bj int) int {
	return commonPrefixLenRunes(s.a[ai:aj:aj], s.b[bi:bj:bj])
}
func (s runesSeqs) commonSuffixLen(ai, aj, bi, bj int) int {
	return commonSuffixLenRunes(s.a[ai:aj:aj], s.b[bi:bj:bj])
}

// TODO(adonovan): optimize these functions using ideas from:
// - https://go.dev/cl/408116 common.go
// - https://go.dev/cl/421435 xor_generic.go

// TODO(adonovan): factor using generics when available,
// but measure performance impact.

// commonPrefixLen* returns the length of the common prefix of a[ai:aj] and b[bi:bj].
func commonPrefixLenBytes(a, b []byte) int {
	n := min(len(a), len(b))
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}
func commonPrefixLenRunes(a, b []rune) int {
	n := min(len(a), len(b))
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}
func commonPrefixLenString(a, b string) int {
	n := min(len(a), len(b))
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

// commonSuffixLen* returns the length of the common suffix of a[ai:aj] and b[bi:bj].
func commonSuffixLenBytes(a, b []byte) int {
	n := min(len(a), len(b))
	i := 0
	for i < n && a[len(a)-1-i] == b[len(b)-1-i] {
		i++
	}
	return i
}
func commonSuffixLenRunes(a, b []rune) int {
	n := min(len(a), len(b))
	i := 0
	for i < n && a[len(a)-1-i] == b[len(b)-1-i] {
		i++
	}
	return i
}
func commonSuffixLenString(a, b string) int {
	n := min(len(a), len(b))
	i := 0
	for i < n && a[len(a)-1-i] == b[len(b)-1-i] {
		i++
	}
	return i
}
