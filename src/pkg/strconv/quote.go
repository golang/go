// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import (
	"bytes"
	"os"
	"strings"
	"unicode"
	"utf8"
)

const lowerhex = "0123456789abcdef"

// Quote returns a double-quoted Go string literal
// representing s.  The returned string s uses Go escape
// sequences (\t, \n, \xFF, \u0100) for control characters
// and non-ASCII characters.
func Quote(s string) string {
	var buf bytes.Buffer
	buf.WriteByte('"')
	for ; len(s) > 0; s = s[1:] {
		switch c := s[0]; {
		case c == '"':
			buf.WriteString(`\"`)
		case c == '\\':
			buf.WriteString(`\\`)
		case ' ' <= c && c <= '~':
			buf.WriteString(string(c))
		case c == '\a':
			buf.WriteString(`\a`)
		case c == '\b':
			buf.WriteString(`\b`)
		case c == '\f':
			buf.WriteString(`\f`)
		case c == '\n':
			buf.WriteString(`\n`)
		case c == '\r':
			buf.WriteString(`\r`)
		case c == '\t':
			buf.WriteString(`\t`)
		case c == '\v':
			buf.WriteString(`\v`)

		case c >= utf8.RuneSelf && utf8.FullRuneInString(s):
			r, size := utf8.DecodeRuneInString(s)
			if r == utf8.RuneError && size == 1 {
				goto EscX
			}
			s = s[size-1:] // next iteration will slice off 1 more
			if r < 0x10000 {
				buf.WriteString(`\u`)
				for j := uint(0); j < 4; j++ {
					buf.WriteByte(lowerhex[(r>>(12-4*j))&0xF])
				}
			} else {
				buf.WriteString(`\U`)
				for j := uint(0); j < 8; j++ {
					buf.WriteByte(lowerhex[(r>>(28-4*j))&0xF])
				}
			}

		default:
		EscX:
			buf.WriteString(`\x`)
			buf.WriteByte(lowerhex[c>>4])
			buf.WriteByte(lowerhex[c&0xF])
		}
	}
	buf.WriteByte('"')
	return buf.String()
}

// CanBackquote returns whether the string s would be
// a valid Go string literal if enclosed in backquotes.
func CanBackquote(s string) bool {
	for i := 0; i < len(s); i++ {
		if (s[i] < ' ' && s[i] != '\t') || s[i] == '`' {
			return false
		}
	}
	return true
}

func unhex(b byte) (v int, ok bool) {
	c := int(b)
	switch {
	case '0' <= c && c <= '9':
		return c - '0', true
	case 'a' <= c && c <= 'f':
		return c - 'a' + 10, true
	case 'A' <= c && c <= 'F':
		return c - 'A' + 10, true
	}
	return
}

// UnquoteChar decodes the first character or byte in the escaped string
// or character literal represented by the string s.
// It returns four values:
//
//	1) value, the decoded Unicode code point or byte value;
//	2) multibyte, a boolean indicating whether the decoded character requires a multibyte UTF-8 representation;
//	3) tail, the remainder of the string after the character; and
//	4) an error that will be nil if the character is syntactically valid.
//
// The second argument, quote, specifies the type of literal being parsed
// and therefore which escaped quote character is permitted.
// If set to a single quote, it permits the sequence \' and disallows unescaped '.
// If set to a double quote, it permits \" and disallows unescaped ".
// If set to zero, it does not permit either escape and allows both quote characters to appear unescaped.
func UnquoteChar(s string, quote byte) (value int, multibyte bool, tail string, err os.Error) {
	// easy cases
	switch c := s[0]; {
	case c == quote && (quote == '\'' || quote == '"'):
		err = os.EINVAL
		return
	case c >= utf8.RuneSelf:
		r, size := utf8.DecodeRuneInString(s)
		return r, true, s[size:], nil
	case c != '\\':
		return int(s[0]), false, s[1:], nil
	}

	// hard case: c is backslash
	if len(s) <= 1 {
		err = os.EINVAL
		return
	}
	c := s[1]
	s = s[2:]

	switch c {
	case 'a':
		value = '\a'
	case 'b':
		value = '\b'
	case 'f':
		value = '\f'
	case 'n':
		value = '\n'
	case 'r':
		value = '\r'
	case 't':
		value = '\t'
	case 'v':
		value = '\v'
	case 'x', 'u', 'U':
		n := 0
		switch c {
		case 'x':
			n = 2
		case 'u':
			n = 4
		case 'U':
			n = 8
		}
		v := 0
		if len(s) < n {
			err = os.EINVAL
			return
		}
		for j := 0; j < n; j++ {
			x, ok := unhex(s[j])
			if !ok {
				err = os.EINVAL
				return
			}
			v = v<<4 | x
		}
		s = s[n:]
		if c == 'x' {
			// single-byte string, possibly not UTF-8
			value = v
			break
		}
		if v > unicode.MaxRune {
			err = os.EINVAL
			return
		}
		value = v
		multibyte = true
	case '0', '1', '2', '3', '4', '5', '6', '7':
		v := int(c) - '0'
		if len(s) < 2 {
			err = os.EINVAL
			return
		}
		for j := 0; j < 2; j++ { // one digit already; two more
			x := int(s[j]) - '0'
			if x < 0 || x > 7 {
				return
			}
			v = (v << 3) | x
		}
		s = s[2:]
		if v > 255 {
			err = os.EINVAL
			return
		}
		value = v
	case '\\':
		value = '\\'
	case '\'', '"':
		if c != quote {
			err = os.EINVAL
			return
		}
		value = int(c)
	default:
		err = os.EINVAL
		return
	}
	tail = s
	return
}

// Unquote interprets s as a single-quoted, double-quoted,
// or backquoted Go string literal, returning the string value
// that s quotes.  (If s is single-quoted, it would be a Go
// character literal; Unquote returns the corresponding
// one-character string.)
func Unquote(s string) (t string, err os.Error) {
	n := len(s)
	if n < 2 {
		return "", os.EINVAL
	}
	quote := s[0]
	if quote != s[n-1] {
		return "", os.EINVAL
	}
	s = s[1 : n-1]

	if quote == '`' {
		if strings.Contains(s, "`") {
			return "", os.EINVAL
		}
		return s, nil
	}
	if quote != '"' && quote != '\'' {
		return "", os.EINVAL
	}

	var buf bytes.Buffer
	for len(s) > 0 {
		c, multibyte, ss, err := UnquoteChar(s, quote)
		if err != nil {
			return "", err
		}
		s = ss
		if c < utf8.RuneSelf || !multibyte {
			buf.WriteByte(byte(c))
		} else {
			buf.WriteString(string(c))
		}
		if quote == '\'' && len(s) != 0 {
			// single-quoted must be single character
			return "", os.EINVAL
		}
	}
	return buf.String(), nil
}
