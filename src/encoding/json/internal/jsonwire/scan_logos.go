// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsonwire

import (
	"encoding/json/internal/jsonflags"
	"internal/jsonscan"
	"unicode/utf16"
	"unicode/utf8"
)

const (
	probeLen = 32
	hotSkip  = 32
)

func skipEscapeHTML(b []byte, n int) (next int, hot bool) {
	k := jsonscan.IndexEscapeHTML(b[n:])
	return n + k, k >= hotSkip
}

func skipEscape(b []byte, n int, flags *jsonflags.Flags) (next int, hot bool) {
	if flags.Get(jsonflags.EscapeForHTML) {
		return skipEscapeHTML(b, n)
	}
	k := jsonscan.IndexEscape(b[n:])
	return n + k, k >= hotSkip
}

func skipStringSpecial(b []byte, n int) (next int, hot bool) {
	k := jsonscan.IndexStringSpecial(b[n:])
	return n + k, k >= hotSkip
}

func needEscapeStep(src []byte, n int) (next int, found bool) {
	if c := src[n]; c < utf8.RuneSelf {
		return n + 1, escapeASCII[c] > 0
	}
	r, rn := utf8.DecodeRune(src[n:])
	return n + rn, r == utf8.RuneError || r == '\u2028' || r == '\u2029'
}

func needEscapeSparse(src []byte) (n int, found bool) {
	n, hot := skipEscapeHTML(src, 0)
	for hot && n < len(src) {
		if n, found = needEscapeStep(src, n); found {
			return n, true
		}
		n, hot = skipEscapeHTML(src, n)
	}
	return n, false
}

func appendQuoteSparse(dst, src []byte, flags *jsonflags.Flags) (out []byte, i, n int, invalid bool) {
	n, hot := skipEscape(src, 0, flags)
	for hot && n < len(src) {
		if c := src[n]; c < utf8.RuneSelf {
			n++
			if escapeASCII[c] != 0 &&
				(!(c == '<' || c == '>' || c == '&') || flags.Get(jsonflags.EscapeForHTML)) {
				dst = append(dst, src[i:n-1]...)
				dst = appendEscapedASCII(dst, c)
				i = n
			}
		} else {
			r, rn := utf8.DecodeRune(src[n:])
			n += rn
			if isInvalidUTF8(r, rn) {
				invalid = true
				dst = append(dst, src[i:n-rn]...)
				dst = append(dst, "\ufffd"...)
				i = n
			} else if (r == '\u2028' || r == '\u2029') && flags.Get(jsonflags.EscapeForJS) {
				dst = append(dst, src[i:n-rn]...)
				dst = appendEscapedUnicode(dst, r)
				i = n
			}
		}
		n, hot = skipEscape(src, n, flags)
	}
	return dst, i, n, invalid
}

func consumeSparse(flags *ValueFlags, b []byte, n int) int {
	n, hot := skipStringSpecial(b, n)
	for hot && n < len(b) {
		c := b[n]
		switch {
		case c == '"':
			return n
		case c == '\\':
			if n+1 >= len(b) {
				return n
			}
			switch e := b[n+1]; e {
			case '"', '\\', 'b', 'f', 'n', 'r', 't':
				flags.Join(stringNonVerbatim)
				n += 2
			case '/':
				flags.Join(stringNonVerbatim | stringNonCanonical)
				n += 2
			case 'u':
				if uint(len(b)) < uint(n+6) {
					return n
				}
				v1, ok := parseHexUint16(b[n+2 : n+6])
				if !ok || utf16.IsSurrogate(rune(v1)) {
					return n
				}
				flags.Join(stringNonVerbatim)
				switch v1 {
				case '\b', '\f', '\n', '\r', '\t':
					flags.Join(stringNonCanonical)
				default:
					if v1 >= ' ' {
						flags.Join(stringNonCanonical)
					} else {
						for _, c := range b[n+2 : n+6] {
							if 'A' <= c && c <= 'F' {
								flags.Join(stringNonCanonical)
							}
						}
					}
				}
				n += 6
			default:
				return n
			}
		case c < utf8.RuneSelf:
			return n
		default:
			if _, rn := utf8.DecodeRune(b[n:]); rn == 1 {
				return n
			} else {
				n += rn
			}
		}
		n, hot = skipStringSpecial(b, n)
	}
	return n
}

func unquoteSparse(dst, src []byte, i, n int) ([]byte, int, int) {
	n, hot := skipStringSpecial(src, n)
	for hot && n < len(src) {
		c := src[n]
		switch {
		case c == '"':
			return dst, i, n
		case c == '\\':
			if n+1 >= len(src) {
				return dst, i, n
			}
			switch e := src[n+1]; e {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
				dst = append(dst, src[i:n]...)
				dst = append(dst, unescaped(e))
				n += 2
				i = n
			case 'u':
				if uint(len(src)) < uint(n+6) {
					return dst, i, n
				}
				v1, ok := parseHexUint16(src[n+2 : n+6])
				if !ok || utf16.IsSurrogate(rune(v1)) {
					return dst, i, n
				}
				dst = append(dst, src[i:n]...)
				dst = utf8.AppendRune(dst, rune(v1))
				n += 6
				i = n
			default:
				return dst, i, n
			}
		case c < utf8.RuneSelf:
			return dst, i, n
		default:
			if _, rn := utf8.DecodeRune(src[n:]); rn == 1 {
				return dst, i, n
			} else {
				n += rn
			}
		}
		n, hot = skipStringSpecial(src, n)
	}
	return dst, i, n
}

func unescaped(c byte) byte {
	switch c {
	case 'b':
		return '\b'
	case 'f':
		return '\f'
	case 'n':
		return '\n'
	case 'r':
		return '\r'
	case 't':
		return '\t'
	}
	return c
}
