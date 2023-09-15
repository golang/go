// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import (
	"fmt"
	"strconv"
	"strings"
)

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

// PrefixToPath is the inverse of PathToPrefix, replacing escape sequences with
// the original character.
func PrefixToPath(s string) (string, error) {
	percent := strings.IndexByte(s, '%')
	if percent == -1 {
		return s, nil
	}

	p := make([]byte, 0, len(s))
	for i := 0; i < len(s); {
		if s[i] != '%' {
			p = append(p, s[i])
			i++
			continue
		}
		if i+2 >= len(s) {
			// Not enough characters remaining to be a valid escape
			// sequence.
			return "", fmt.Errorf("malformed prefix %q: escape sequence must contain two hex digits", s)
		}

		b, err := strconv.ParseUint(s[i+1:i+3], 16, 8)
		if err != nil {
			// Not a valid escape sequence.
			return "", fmt.Errorf("malformed prefix %q: escape sequence %q must contain two hex digits", s, s[i:i+3])
		}

		p = append(p, byte(b))
		i += 3
	}
	return string(p), nil
}
