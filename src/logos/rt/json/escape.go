// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"internal/jsonscan"
	"unicode/utf8"
)

const hexDigits = "0123456789abcdef"

const (
	multiClean = iota
	multiInvalid
	multiSeparator
	multiKinds
)

type escapeUnitBytes [6]byte

var (
	escapeExtra      = buildASCIIExtra(plainUnitLen)
	quotedExtra      = buildASCIIExtra(quotedUnitLen)
	multiEscapeExtra = buildMultiExtra(plainUnitLen)
	multiQuotedExtra = buildMultiExtra(quotedUnitLen)
)

func plainUnitLen(_ escapeUnitBytes, n int) int {
	return n
}

func quotedUnitLen(u escapeUnitBytes, n int) int {
	out := n
	for _, b := range u[:n] {
		if b == '\\' || b == '"' {
			out++
		}
	}
	return out
}

func asciiNeedsEscape(c byte) bool {
	return c < 0x20 || c == '"' || c == '\\' || c == '<' || c == '>' || c == '&'
}

func buildASCIIExtra(unitLen func(escapeUnitBytes, int) int) [utf8.RuneSelf]uint8 {
	var t [utf8.RuneSelf]uint8
	for c := range utf8.RuneSelf {
		if asciiNeedsEscape(byte(c)) {
			u, n := asciiUnit(byte(c))
			t[c] = uint8(unitLen(u, n) - 1)
		}
	}
	return t
}

func buildMultiExtra(unitLen func(escapeUnitBytes, int) int) [multiKinds]int {
	var t [multiKinds]int
	var invalid escapeUnitBytes
	n := copy(invalid[:], invalidUTF8Escape)
	t[multiInvalid] = unitLen(invalid, n) - 1
	sep, n := separatorUnit('\u2028')
	t[multiSeparator] = unitLen(sep, n) - len("\u2028")
	return t
}

func asciiUnit(c byte) (escapeUnitBytes, int) {
	switch c {
	case '"', '\\':
		return escapeUnitBytes{'\\', c}, 2
	case '\b':
		return escapeUnitBytes{'\\', 'b'}, 2
	case '\f':
		return escapeUnitBytes{'\\', 'f'}, 2
	case '\n':
		return escapeUnitBytes{'\\', 'n'}, 2
	case '\r':
		return escapeUnitBytes{'\\', 'r'}, 2
	case '\t':
		return escapeUnitBytes{'\\', 't'}, 2
	}
	return escapeUnitBytes{'\\', 'u', '0', '0', hexDigits[c>>4], hexDigits[c&0xf]}, 6
}

func separatorUnit(r rune) (escapeUnitBytes, int) {
	return escapeUnitBytes{'\\', 'u', '2', '0', '2', hexDigits[r&0xf]}, 6
}

func classifyMulti(s string, i int) (kind int, r rune, size int) {
	r, size = utf8.DecodeRuneInString(s[i:])
	switch {
	case r == utf8.RuneError && size == 1:
		return multiInvalid, r, size
	case r == '\u2028' || r == '\u2029':
		return multiSeparator, r, size
	}
	return multiClean, r, size
}

const kernelEntryRun = 16

func skipPlain(s string, i int) int {
	return i + jsonscan.IndexEscapeHTML(s[i:])
}

func firstEscape(s string, i int) int {
	run := 0
	for i < len(s) {
		c := s[i]
		if c < utf8.RuneSelf {
			if escapeExtra[c] != 0 {
				return i
			}
			i++
			if run++; run == kernelEntryRun {
				i, run = skipPlain(s, i), 0
			}
			continue
		}
		kind, _, size := classifyMulti(s, i)
		if kind != multiClean {
			return i
		}
		i += size
		run = 0
	}
	return i
}

func escapeUnit(s string, i int) (u escapeUnitBytes, n, size int) {
	if c := s[i]; c < utf8.RuneSelf {
		u, n = asciiUnit(c)
		return u, n, 1
	}
	kind, r, size := classifyMulti(s, i)
	if kind == multiInvalid {
		n = copy(u[:], invalidUTF8Escape)
		return u, n, size
	}
	u, n = separatorUnit(r)
	return u, n, size
}

func appendDoubled(dst []byte, unit []byte) []byte {
	for _, b := range unit {
		if b == '\\' || b == '"' {
			dst = append(dst, '\\')
		}
		dst = append(dst, b)
	}
	return dst
}

// AppendEscaped appends the escaped body of the JSON string s to dst,
// without the surrounding quotes, exactly as encoding/json writes it:
// <, > and & become \u003c, \u003e and \u0026, U+2028 and U+2029 become
// \u2028 and \u2029, control characters become \b, \f, \n, \r, \t or \u00XX,
// the quote and the backslash are backslashed, 0x7f is left alone and an
// invalid UTF-8 byte follows the jsonv2 experiment rule. A string that
// needs no escape is appended with a single copy.
func AppendEscaped(dst []byte, s string) []byte {
	i := firstEscape(s, 0)
	if i == len(s) {
		return append(dst, s...)
	}
	start := 0
	for i < len(s) {
		dst = append(dst, s[start:i]...)
		u, n, size := escapeUnit(s, i)
		dst = append(dst, u[:n]...)
		i += size
		start = i
		i = firstEscape(s, i)
	}
	return append(dst, s[start:]...)
}

// EscapedLen returns the exact length of AppendEscaped(nil, s).
func EscapedLen(s string) int {
	n, run := len(s), 0
	for i := 0; i < len(s); {
		if c := s[i]; c < utf8.RuneSelf {
			i++
			if extra := escapeExtra[c]; extra != 0 {
				n += int(extra)
				run = 0
				continue
			}
			if run++; run == kernelEntryRun {
				i, run = skipPlain(s, i), 0
			}
			continue
		}
		kind, _, size := classifyMulti(s, i)
		n += multiEscapeExtra[kind]
		i += size
		run = 0
	}
	return n
}

// AppendQuoted appends to dst the body, without outer quotes, of the JSON
// string that the ,string option produces for a string field: the inner
// value is a quote, the HTML-escaped s and a quote, and that whole inner
// value is escaped once more with HTML escaping off. For s = `a<` the
// result is `\"a\\u003c\"`.
func AppendQuoted(dst []byte, s string) []byte {
	dst = append(dst, '\\', '"')
	start := 0
	for i := firstEscape(s, 0); i < len(s); i = firstEscape(s, i) {
		dst = append(dst, s[start:i]...)
		u, n, size := escapeUnit(s, i)
		dst = appendDoubled(dst, u[:n])
		i += size
		start = i
	}
	dst = append(dst, s[start:]...)
	return append(dst, '\\', '"')
}

// QuotedLen returns the exact length of AppendQuoted(nil, s).
func QuotedLen(s string) int {
	n, run := len(s)+len(`\"\"`), 0
	for i := 0; i < len(s); {
		if c := s[i]; c < utf8.RuneSelf {
			i++
			if extra := quotedExtra[c]; extra != 0 {
				n += int(extra)
				run = 0
				continue
			}
			if run++; run == kernelEntryRun {
				i, run = skipPlain(s, i), 0
			}
			continue
		}
		kind, _, size := classifyMulti(s, i)
		n += multiQuotedExtra[kind]
		i += size
		run = 0
	}
	return n
}
