// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import "internal/cpu"

//go:noescape

// indexShortStr returns the index of the first instance of c in s, or -1 if c is not present in s.
// indexShortStr requires 2 <= len(c) <= shortStringLen
func indexShortStr(s, c string) int  // ../runtime/asm_amd64.s
func countByte(s string, c byte) int // ../runtime/asm_amd64.s

var shortStringLen int

func init() {
	if cpu.X86.HasAVX2 {
		shortStringLen = 63
	} else {
		shortStringLen = 31
	}
}

// Index returns the index of the first instance of substr in s, or -1 if substr is not present in s.
func Index(s, substr string) int {
	n := len(substr)
	switch {
	case n == 0:
		return 0
	case n == 1:
		return IndexByte(s, substr[0])
	case n == len(s):
		if substr == s {
			return 0
		}
		return -1
	case n > len(s):
		return -1
	case n <= shortStringLen:
		// Use brute force when s and substr both are small
		if len(s) <= 64 {
			return indexShortStr(s, substr)
		}
		c := substr[0]
		i := 0
		t := s[:len(s)-n+1]
		fails := 0
		for i < len(t) {
			if t[i] != c {
				// IndexByte skips 16/32 bytes per iteration,
				// so it's faster than indexShortStr.
				o := IndexByte(t[i:], c)
				if o < 0 {
					return -1
				}
				i += o
			}
			if s[i:i+n] == substr {
				return i
			}
			fails++
			i++
			// Switch to indexShortStr when IndexByte produces too many false positives.
			// Too many means more that 1 error per 8 characters.
			// Allow some errors in the beginning.
			if fails > (i+16)/8 {
				r := indexShortStr(s[i:], substr)
				if r >= 0 {
					return r + i
				}
				return -1
			}
		}
		return -1
	}
	// Rabin-Karp search
	hashss, pow := hashStr(substr)
	var h uint32
	for i := 0; i < n; i++ {
		h = h*primeRK + uint32(s[i])
	}
	if h == hashss && s[:n] == substr {
		return 0
	}
	for i := n; i < len(s); {
		h *= primeRK
		h += uint32(s[i])
		h -= pow * uint32(s[i-n])
		i++
		if h == hashss && s[i-n:i] == substr {
			return i - n
		}
	}
	return -1
}

// Count counts the number of non-overlapping instances of substr in s.
// If substr is an empty string, Count returns 1 + the number of Unicode code points in s.
func Count(s, substr string) int {
	if len(substr) == 1 && cpu.X86.HasPOPCNT {
		return countByte(s, byte(substr[0]))
	}
	return countGeneric(s, substr)
}
