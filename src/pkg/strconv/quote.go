// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import (
	"os";
	"unicode";
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
	for ; len(s) > 0; s = s[1:len(s)] {
		switch c := s[0]; {
		case c == '"':
			t += `\"`;
		case c == '\\':
			t += `\\`;
		case ' ' <= c && c <= '~':
			t += string(c);
		case c == '\a':
			t += `\a`;
		case c == '\b':
			t += `\b`;
		case c == '\f':
			t += `\f`;
		case c == '\n':
			t += `\n`;
		case c == '\r':
			t += `\r`;
		case c == '\t':
			t += `\t`;
		case c == '\v':
			t += `\v`;

		case c < utf8.RuneSelf:
			t += `\x` + string(lowerhex[c>>4]) + string(lowerhex[c&0xF]);

		case utf8.FullRuneInString(s):
			r, size := utf8.DecodeRuneInString(s);
			if r == utf8.RuneError && size == 1 {
				goto EscX;
			}
			s = s[size-1:len(s)];	// next iteration will slice off 1 more
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
			t += string(lowerhex[c>>4]);
			t += string(lowerhex[c&0xF]);
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

// UnquoteChar decodes the first character or byte in the escaped string
// or character literal represented by the string s.
// It returns four values: 
// 1) value, the decoded Unicode code point or byte value;
// 2) multibyte, a boolean indicating whether the decoded character
//    requires a multibyte UTF-8 representation;
// 3) tail, the remainder of the string after the character; and
// 4) an error that will be nil if the character is syntactically valid.
// The second argument, quote, specifies the type of literal being parsed
// and therefore which escaped quote character is permitted.
// If set to a single quote, it permits the sequence \' and disallows unescaped '.
// If set to a double quote, it permits \" and disallows unescaped ".
// If set to zero, it does not permit either escape and allows both quote characters to appear unescaped.
func UnquoteChar(s string, quote byte) (value int, multibyte bool, tail string, err os.Error) {
	// easy cases
	switch c := s[0]; {
	case c == quote && (quote == '\'' || quote == '"'):
		err = os.EINVAL;
		return;
	case c >= utf8.RuneSelf:
		r, size := utf8.DecodeRuneInString(s);
		return r, true, s[size:len(s)], nil;
	case c != '\\':
		return int(s[0]), false, s[1:len(s)], nil;
	}

	// hard case: c is backslash
	if len(s) <= 1 {
		err = os.EINVAL;
		return;
	}
	c := s[1];
	s = s[2:len(s)];

	switch c {
	case 'a':
		value = '\a';
	case 'b':
		value = '\b';
	case 'f':
		value = '\f';
	case 'n':
		value = '\n';
	case 'r':
		value = '\r';
	case 't':
		value = '\t';
	case 'v':
		value = '\v';
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
		if len(s) < n {
			err = os.EINVAL;
			return;
		}
		for j := 0; j < n; j++ {
			x, ok := unhex(s[j]);
			if !ok {
				err = os.EINVAL;
				return;
			}
			v = v<<4 | x;
		}
		s = s[n:len(s)];
		if c == 'x' {
			// single-byte string, possibly not UTF-8
			value = v;
			break;
		}
		if v > unicode.MaxRune {
			err = os.EINVAL;
			return;
		}
		value = v;
		multibyte = true;
	case '0', '1', '2', '3', '4', '5', '6', '7':
		v := int(c) - '0';
		if len(s) < 2 {
			err = os.EINVAL;
			return;
		}
		for j := 0; j < 2; j++ {	// one digit already; two more
			x := int(s[j]) - '0';
			if x < 0 || x > 7 {
				return;
			}
			v = (v<<3) | x;
		}
		s = s[2:len(s)];
		if v > 255 {
			err = os.EINVAL;
			return;
		}
		value = v;
	case '\\':
		value = '\\';
	case '\'', '"':
		if c != quote {
			err = os.EINVAL;
			return;
		}
		value = int(c);
	default:
		err = os.EINVAL;
		return;
	}
	tail = s;
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
	if n < 2 {
		return;
	}
	quote := s[0];
	if quote != s[n-1] {
		return;
	}
	s = s[1:n-1];

	if quote == '`' {
		return s, nil;
	}
	if quote != '"' && quote != '\'' {
		return;
	}

	// TODO(rsc): String accumulation could be more efficient.
	var tt string;
	for len(s) > 0 {
		c, multibyte, ss, err1 := UnquoteChar(s, quote);
		if err1 != nil {
			err = err1;
			return;
		}
		s = ss;
		if multibyte || c < utf8.RuneSelf {
			tt += string(c);
		} else {
			tt += string([]byte{byte(c)});
		}
		if quote == '\'' && len(s) != 0 {
			// single-quoted must be single character
			return;
		}
	}
	return tt, nil
}
