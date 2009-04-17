// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import (
	"os";
	"utf8";
)

const lowerhex = "0123456789abcdef"

// Quote returns a double-quoted Go string literal
// representing s.  The returned string s uses Go escape
// sequences (\t, \n, \xFF, \u0100) for control characters
// and non-ASCII characters.
func Quote(s string) string {
	// TODO(rsc): String accumulation could be more efficient.
	t := `"`;
	for i := 0; i < len(s); i++ {
		switch {
		case s[i] == '"':
			t += `\"`;
		case s[i] == '\\':
			t += `\\`;
		case ' ' <= s[i] && s[i] <= '~':
			t += string(s[i]);
		case s[i] == '\a':
			t += `\a`;
		case s[i] == '\b':
			t += `\b`;
		case s[i] == '\f':
			t += `\f`;
		case s[i] == '\n':
			t += `\n`;
		case s[i] == '\r':
			t += `\r`;
		case s[i] == '\t':
			t += `\t`;
		case s[i] == '\v':
			t += `\v`;

		case s[i] < utf8.RuneSelf:
			t += `\x` + string(lowerhex[s[i]>>4]) + string(lowerhex[s[i]&0xF]);

		case utf8.FullRuneInString(s, i):
			r, size := utf8.DecodeRuneInString(s, i);
			if r == utf8.RuneError && size == 1 {
				goto EscX;
			}
			i += size-1;  // i++ on next iteration
			if r < 0x10000 {
				t += `\u`;
				for j:=uint(0); j<4; j++ {
					t += string(lowerhex[(r>>(12-4*j))&0xF]);
				}
			} else {
				t += `\U`;
				for j:=uint(0); j<8; j++ {
					t += string(lowerhex[(r>>(28-4*j))&0xF]);
				}
			}

		default:
		EscX:
			t += `\x`;
			t += string(lowerhex[s[i]>>4]);
			t += string(lowerhex[s[i]&0xF]);
		}
	}
	t += `"`;
	return t;
}

// CanBackquote returns whether the string s would be
// a valid Go string literal if enclosed in backquotes.
func CanBackquote(s string) bool {
	for i := 0; i < len(s); i++ {
		if (s[i] < ' ' && s[i] != '\t') || s[i] == '`' {
			return false;
		}
	}
	return true;
}

func unhex(b byte) (v int, ok bool) {
	c := int(b);
	switch {
	case '0' <= c && c <= '9':
		return c - '0', true;
	case 'a' <= c && c <= 'f':
		return c - 'a' + 10, true;
	case 'A' <= c && c <= 'F':
		return c - 'A' + 10, true;
	}
	return;
}

func unquoteChar(s string, i int, q byte) (t string, ii int, err os.Error) {
	err = os.EINVAL;  // assume error for easy return

	// easy cases
	switch c := s[i]; {
	case c >= utf8.RuneSelf:
		r, size := utf8.DecodeRuneInString(s, i);
		return s[i:i+size], i+size, nil;
	case c == q:
		return;
	case c != '\\':
		return s[i:i+1], i+1, nil;
	}

	// hard case: c is backslash
	if i+1 >= len(s) {
		return;
	}
	c := s[i+1];
	i += 2;

	switch c {
	case 'a':
		return "\a", i, nil;
	case 'b':
		return "\b", i, nil;
	case 'f':
		return "\f", i, nil;
	case 'n':
		return "\n", i, nil;
	case 'r':
		return "\r", i, nil;
	case 't':
		return "\t", i, nil;
	case 'v':
		return "\v", i, nil;
	case 'x', 'u', 'U':
		n := 0;
		switch c {
		case 'x':
			n = 2;
		case 'u':
			n = 4;
		case 'U':
			n = 8;
		}
		v := 0;
		for j := 0; j < n; j++ {
			if i+j >= len(s) {
				return;
			}
			x, ok := unhex(s[i+j]);
			if !ok {
				return;
			}
			v = v<<4 | x;
		}
		if c == 'x' {
			return string([]byte{byte(v)}), i+n, nil;
		}
		if v > utf8.RuneMax {
			return;
		}
		return string(v), i+n, nil;
	case '0', '1', '2', '3', '4', '5', '6', '7':
		v := 0;
		i--;
		for j := 0; j < 3; j++ {
			if i+j >= len(s) {
				return;
			}
			x := int(s[i+j]) - '0';
			if x < 0 || x > 7 {
				return;
			}
			v = (v<<3) | x;
		}
		if v > 255 {
			return;
		}
		return string(v), i+3, nil;
			
	case '\\', q:
		return string(c), i, nil;
	}
	return;
}

// Unquote interprets s as a single-quoted, double-quoted,
// or backquoted Go string literal, returning the string value
// that s quotes.  (If s is single-quoted, it would be a Go
// character literal; Unquote returns the corresponding
// one-character string.)
func Unquote(s string) (t string, err os.Error) {
	err = os.EINVAL;  // assume error for easy return
	n := len(s);
	if n < 2 || s[0] != s[n-1] {
		return;
	}

	switch s[0] {
	case '`':
		t := s[1:n-1];
		return t, nil;

	case '"', '\'':
		// TODO(rsc): String accumulation could be more efficient.
		t := "";
		q := s[0];
		var c string;
		var err os.Error;
		for i := 1; i < n-1; {
			c, i, err = unquoteChar(s, i, q);
			if err != nil {
				return "", err;
			}
			t += c;
			if q == '\'' && i != n-1 {
				// single-quoted must be single character
				return;
			}
			if i > n-1 {
				// read too far
				return;
			}
		}
		return t, nil
	}
	return;
}
