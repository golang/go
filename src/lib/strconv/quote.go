// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import (
	"utf8";
)

const lowerhex = "0123456789abcdef"

// Quote returns a double-quoted Go string literal
// representing s.  The returned string s uses Go escape
// sequences (\t, \n, \xFF, \u0100) for control characters
// and non-ASCII characters.
func Quote(s string) string {
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
		if s[i] < ' ' || s[i] == '`' {
			return false;
		}
	}
	return true;
}

