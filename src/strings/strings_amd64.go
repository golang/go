// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

//go:noescape

// indexShortStr returns the index of the first instance of c in s, or -1 if c is not present in s.
// indexShortStr requires 2 <= len(c) <= shortStringLen
func indexShortStr(s, c string) int // ../runtime/asm_$GOARCH.s
func supportAVX2() bool             // ../runtime/asm_$GOARCH.s

var shortStringLen int

func init() {
	if supportAVX2() {
		shortStringLen = 63
	} else {
		shortStringLen = 31
	}
}

// Index returns the index of the first instance of sep in s, or -1 if sep is not present in s.
func Index(s, sep string) int {
	n := len(sep)
	switch {
	case n == 0:
		return 0
	case n == 1:
		return IndexByte(s, sep[0])
	case n == len(s):
		if sep == s {
			return 0
		}
		return -1
	case n > len(s):
		return -1
	case n <= shortStringLen:
		// Use brute force when s and sep both are small
		if len(s) <= 64 {
			return indexShortStr(s, sep)
		}
		c := sep[0]
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
			if s[i:i+n] == sep {
				return i
			}
			fails++
			i++
			// Switch to indexShortStr when IndexByte produces too many false positives.
			// Too many means more that 1 error per 8 characters.
			// Allow some errors in the beginning.
			if fails > (i+16)/8 {
				r := indexShortStr(s[i:], sep)
				if r >= 0 {
					return r + i
				}
				return -1
			}
		}
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
