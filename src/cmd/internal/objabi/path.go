// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import "strings"

// PathToPrefix converts raw string to the prefix that will be used in the
// symbol table. All control characters, space, '%' and '"', as well as
// non-7-bit clean bytes turn into %xx. The period needs escaping only in the
// last segment of the path, and it makes for happier users if we escape that as
// little as possible.
func PathToPrefix(s string) string {
	slash := strings.LastIndex(s, "/")
	// check for chars that need escaping
	n := 0
	for r := 0; r < len(s); r++ {
		if c := s[r]; c <= ' ' || (c == '.' && r > slash) || c == '%' || c == '"' || c >= 0x7F {
			n++
		}
	}

	// quick exit
	if n == 0 {
		return s
	}

	// escape
	const hex = "0123456789abcdef"
	p := make([]byte, 0, len(s)+2*n)
	for r := 0; r < len(s); r++ {
		if c := s[r]; c <= ' ' || (c == '.' && r > slash) || c == '%' || c == '"' || c >= 0x7F {
			p = append(p, '%', hex[c>>4], hex[c&0xF])
		} else {
			p = append(p, c)
		}
	}

	return string(p)
}
