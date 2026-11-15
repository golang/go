// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"unicode"
	"unicode/utf8"
	"unsafe"
)

var simpleEscape = buildSimpleEscape()

func buildSimpleEscape() [256]byte {
	var t [256]byte
	t['"'] = '"'
	t['\\'] = '\\'
	t['/'] = '/'
	t['\''] = '\''
	t['b'] = '\b'
	t['f'] = '\f'
	t['n'] = '\n'
	t['r'] = '\r'
	t['t'] = '\t'
	return t
}

func unsafeString(b []byte) string {
	return unsafe.String(unsafe.SliceData(b), len(b))
}

func isSurrogate(r rune) bool {
	return 0xd800 <= r && r < 0xe000
}

func decodeSurrogates(r1, r2 rune) rune {
	if 0xd800 <= r1 && r1 < 0xdc00 && 0xdc00 <= r2 && r2 < 0xe000 {
		return (r1-0xd800)<<10 | (r2 - 0xdc00) + 0x10000
	}
	return unicode.ReplacementChar
}

func appendUnicodeEscape(dst, body []byte, i int) ([]byte, int, bool) {
	r := getu4(body[i:])
	if r < 0 {
		return dst, i, false
	}
	i += 6
	if isSurrogate(r) {
		if dec := decodeSurrogates(r, getu4(body[i:])); dec != unicode.ReplacementChar {
			return utf8.AppendRune(dst, dec), i + 6, true
		}
		r = unicode.ReplacementChar
	}
	return utf8.AppendRune(dst, r), i, true
}

func appendEscape(dst, body []byte, i int) ([]byte, int, bool) {
	if i+1 >= len(body) {
		return dst, i, false
	}
	c := body[i+1]
	if r := simpleEscape[c]; r != 0 {
		return append(dst, r), i + 2, true
	}
	if c != 'u' {
		return dst, i, false
	}
	return appendUnicodeEscape(dst, body, i)
}

func appendUnquoted(dst, body []byte) ([]byte, bool) {
	run := 0
	for i := 0; i < len(body); {
		c := body[i]
		switch {
		case c == '\\':
			var ok bool
			if dst, i, ok = appendEscape(dst, body, i); !ok {
				return dst, false
			}
			run = 0
		case c == '"' || c < ' ':
			return dst, false
		case c < utf8.RuneSelf:
			dst = append(dst, c)
			i++
			if run++; run == kernelEntryRun {
				end := skipPlainBytes(body, i)
				dst = append(dst, body[i:end]...)
				i, run = end, 0
			}
		default:
			r, size := utf8.DecodeRune(body[i:])
			dst = utf8.AppendRune(dst, r)
			i += size
			run = 0
		}
	}
	return dst, true
}

func unquoteLiteral(item string) (string, bool) {
	if len(item) < 2 || item[0] != '"' || item[len(item)-1] != '"' {
		return "", false
	}
	body := []byte(item[1 : len(item)-1])
	out, ok := appendUnquoted(make([]byte, 0, len(body)), body)
	if !ok {
		return "", false
	}
	return unsafeString(out), true
}
