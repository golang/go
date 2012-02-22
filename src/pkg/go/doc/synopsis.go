// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import "unicode"

// firstSentenceLen returns the length of the first sentence in s.
// The sentence ends after the first period followed by space and
// not preceded by exactly one uppercase letter.
//
func firstSentenceLen(s string) int {
	var ppp, pp, p rune
	for i, q := range s {
		if q == '\n' || q == '\r' || q == '\t' {
			q = ' '
		}
		if q == ' ' && p == '.' && (!unicode.IsUpper(pp) || unicode.IsUpper(ppp)) {
			return i
		}
		ppp, pp, p = pp, p, q
	}
	return len(s)
}

// Synopsis returns a cleaned version of the first sentence in s.
// That sentence ends after the first period followed by space and
// not preceded by exactly one uppercase letter. The result string
// has no \n, \r, or \t characters and uses only single spaces between
// words.
//
func Synopsis(s string) string {
	n := firstSentenceLen(s)
	var b []byte
	p := byte(' ')
	for i := 0; i < n; i++ {
		q := s[i]
		if q == '\n' || q == '\r' || q == '\t' {
			q = ' '
		}
		if q != ' ' || p != ' ' {
			b = append(b, q)
			p = q
		}
	}
	// remove trailing blank, if any
	if n := len(b); n > 0 && p == ' ' {
		b = b[0 : n-1]
	}
	return string(b)
}
