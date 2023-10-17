// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test UTF-8 in strings and character constants.

package main

import "unicode/utf8"

func main() {
	var chars [6]rune
	chars[0] = 'a'
	chars[1] = 'b'
	chars[2] = 'c'
	chars[3] = '\u65e5'
	chars[4] = '\u672c'
	chars[5] = '\u8a9e'
	s := ""
	for i := 0; i < 6; i++ {
		s += string(chars[i])
	}
	var l = len(s)
	for w, i, j := 0, 0, 0; i < l; i += w {
		var r rune
		r, w = utf8.DecodeRuneInString(s[i:len(s)])
		if w == 0 {
			panic("zero width in string")
		}
		if r != chars[j] {
			panic("wrong value from string")
		}
		j++
	}
	// encoded as bytes:  'a' 'b' 'c' e6 97 a5 e6 9c ac e8 aa 9e
	const L = 12
	if L != l {
		panic("wrong length constructing array")
	}
	a := make([]byte, L)
	a[0] = 'a'
	a[1] = 'b'
	a[2] = 'c'
	a[3] = 0xe6
	a[4] = 0x97
	a[5] = 0xa5
	a[6] = 0xe6
	a[7] = 0x9c
	a[8] = 0xac
	a[9] = 0xe8
	a[10] = 0xaa
	a[11] = 0x9e
	for w, i, j := 0, 0, 0; i < L; i += w {
		var r rune
		r, w = utf8.DecodeRune(a[i:L])
		if w == 0 {
			panic("zero width in bytes")
		}
		if r != chars[j] {
			panic("wrong value from bytes")
		}
		j++
	}
}
