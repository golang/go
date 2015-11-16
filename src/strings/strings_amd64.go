// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

// indexShortStr returns the index of the first instance of c in s, or -1 if c is not present in s.
// indexShortStr requires 2 <= len(c) <= shortStringLen
func indexShortStr(s, c string) int // ../runtime/asm_$GOARCH.s
const shortStringLen = 31

// Index returns the index of the first instance of sep in s, or -1 if sep is not present in s.
func Index(s, sep string) int {
	n := len(sep)
	switch {
	case n == 0:
		return 0
	case n == 1:
		return IndexByte(s, sep[0])
	case n <= shortStringLen:
		return indexShortStr(s, sep)
	case n == len(s):
		if sep == s {
			return 0
		}
		return -1
	case n > len(s):
		return -1
	}
	// Rabin-Karp search
	hashsep, pow := hashStr(sep)
	var h uint32
	for i := 0; i < n; i++ {
		h = h*primeRK + uint32(s[i])
	}
	if h == hashsep && s[:n] == sep {
		return 0
	}
	for i := n; i < len(s); {
		h *= primeRK
		h += uint32(s[i])
		h -= pow * uint32(s[i-n])
		i++
		if h == hashsep && s[i-n:i] == sep {
			return i - n
		}
	}
	return -1
}
