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

const sparseSkip = 32

func skipPlain(s string, i int) (int, bool) {
	if len(s)-i < sparseSkip {
		return i, false
	}
	end := i + jsonscan.IndexEscapeHTML(s[i:])
	return end, end-i >= sparseSkip
}

func extraAt(s string, i int, ascii *[utf8.RuneSelf]uint8, multi *[multiKinds]int) (int, int) {
	if c := s[i]; c < utf8.RuneSelf {
		return int(ascii[c]), 1
	}
	kind, _, size := classifyMulti(s, i)
	return multi[kind], size
}

func extraSparse(s string, ascii *[utf8.RuneSelf]uint8, multi *[multiKinds]int) (int, int) {
	n := 0
	i, sparse := skipPlain(s, 0)
	for sparse && i < len(s) {
		extra, size := extraAt(s, i, ascii, multi)
		n += extra
		i, sparse = skipPlain(s, i+size)
	}
	return n, i
}

func extraLen(s string, ascii *[utf8.RuneSelf]uint8, multi *[multiKinds]int) int {
	n, i := 0, 0
	if len(s) >= sparseSkip {
		n, i = extraSparse(s, ascii, multi)
	}
	for i < len(s) {
		if c := s[i]; c < utf8.RuneSelf {
			n += int(ascii[c])
			i++
			continue
		}
		kind, _, size := classifyMulti(s, i)
		n += multi[kind]
		i += size
	}
	return n
}

func firstEscapeSparse(s string, i int) int {
	i, sparse := skipPlain(s, i)
	for sparse && i < len(s) {
		extra, size := extraAt(s, i, &escapeExtra, &multiEscapeExtra)
		if extra != 0 {
			return i
		}
		i, sparse = skipPlain(s, i+size)
	}
	return i
}

func firstEscape(s string, i, prevRun int) int {
	if prevRun >= sparseSkip {
		i = firstEscapeSparse(s, i)
	}
	for i < len(s) {
		c := s[i]
		if c < utf8.RuneSelf {
			if escapeExtra[c] != 0 {
				return i
			}
			i++
			continue
		}
		kind, _, size := classifyMulti(s, i)
		if kind != multiClean {
			return i
		}
		i += size
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
	i := firstEscape(s, 0, len(s))
	if i == len(s) {
		return append(dst, s...)
	}
	start := 0
	for i < len(s) {
		run := i - start
		dst = append(dst, s[start:i]...)
		u, n, size := escapeUnit(s, i)
		dst = append(dst, u[:n]...)
		i += size
		start = i
		i = firstEscape(s, i, run)
	}
	return append(dst, s[start:]...)
}

// EscapedLen returns the exact length of AppendEscaped(nil, s).
func EscapedLen(s string) int {
	return len(s) + extraLen(s, &escapeExtra, &multiEscapeExtra)
}

// AppendQuoted appends to dst the body, without outer quotes, of the JSON
// string that the ,string option produces for a string field: the inner
// value is a quote, the HTML-escaped s and a quote, and that whole inner
// value is escaped once more with HTML escaping off. For s = `a<` the
// result is `\"a\\u003c\"`.
func AppendQuoted(dst []byte, s string) []byte {
	dst = append(dst, '\\', '"')
	start := 0
	i := firstEscape(s, 0, len(s))
	for i < len(s) {
		run := i - start
		dst = append(dst, s[start:i]...)
		u, n, size := escapeUnit(s, i)
		dst = appendDoubled(dst, u[:n])
		i += size
		start = i
		i = firstEscape(s, i, run)
	}
	dst = append(dst, s[start:]...)
	return append(dst, '\\', '"')
}

// QuotedLen returns the exact length of AppendQuoted(nil, s).
func QuotedLen(s string) int {
	return len(s) + len(`\"\"`) + extraLen(s, &quotedExtra, &multiQuotedExtra)
}
