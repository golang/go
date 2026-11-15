// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"internal/jsonscan"
	"unicode/utf8"
)

const maxDepth = 10000

const (
	classPlain = iota
	classQuote
	classBackslash
	classControl
	classHigh
)

const (
	strEscaped = 1 << iota
	strHigh
)

var (
	stringClass = buildStringClass()
	valueStart  = buildValueStart()
)

func buildStringClass() [256]uint8 {
	var t [256]uint8
	for c := range 256 {
		switch {
		case c == '"':
			t[c] = classQuote
		case c == '\\':
			t[c] = classBackslash
		case c < 0x20:
			t[c] = classControl
		case c >= utf8.RuneSelf:
			t[c] = classHigh
		}
	}
	return t
}

func buildValueStart() [256]byte {
	var t [256]byte
	for _, c := range []byte(`"{[tfn`) {
		t[c] = c
	}
	t['-'] = '0'
	for c := '0'; c <= '9'; c++ {
		t[c] = '0'
	}
	return t
}

func skipSpace(data []byte, i int) int {
	for i < len(data) {
		switch data[i] {
		case ' ', '\t', '\r', '\n':
			i++
		default:
			return i
		}
	}
	return i
}

func isDigit(c byte) bool {
	return '0' <= c && c <= '9'
}

func isNumberStart(c byte) bool {
	return c == '-' || isDigit(c)
}

func isDelimiter(c byte) bool {
	switch c {
	case ' ', '\t', '\r', '\n', ',', '}', ']':
		return true
	}
	return false
}

func skipDigits(data []byte, i int) int {
	for i < len(data) && isDigit(data[i]) {
		i++
	}
	return i
}

func hexVal(c byte) rune {
	switch {
	case '0' <= c && c <= '9':
		return rune(c - '0')
	case 'a' <= c && c <= 'f':
		return rune(c-'a') + 10
	case 'A' <= c && c <= 'F':
		return rune(c-'A') + 10
	}
	return -1
}

func getu4(b []byte) rune {
	if len(b) < 6 || b[0] != '\\' || b[1] != 'u' {
		return -1
	}
	var r rune
	for _, c := range b[2:6] {
		h := hexVal(c)
		if h < 0 {
			return -1
		}
		r = r<<4 | h
	}
	return r
}

func skipEscape(data []byte, j int) int {
	if j+1 >= len(data) {
		return -1
	}
	switch data[j+1] {
	case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
		return j + 1
	case 'u':
		if getu4(data[j:]) >= 0 {
			return j + 5
		}
	}
	return -1
}

func skipPlainBytes(data []byte, i int) int {
	return i + jsonscan.IndexStringSpecial(data[i:])
}

func scanString(data []byte, i int) (end, flags int) {
	run := 0
	for j := i + 1; j < len(data); j++ {
		class := stringClass[data[j]]
		if class == classPlain {
			if run++; run == kernelEntryRun {
				j, run = skipPlainBytes(data, j+1)-1, 0
			}
			continue
		}
		run = 0
		switch class {
		case classQuote:
			return j + 1, flags
		case classBackslash:
			flags |= strEscaped
			if j = skipEscape(data, j); j < 0 {
				return -1, 0
			}
		case classControl:
			return -1, 0
		case classHigh:
			flags |= strHigh
		}
	}
	return -1, 0
}

func plainBody(body []byte, flags int) bool {
	return flags == 0 || (flags == strHigh && utf8.Valid(body))
}

func scanIntPart(data []byte, i int) int {
	if i >= len(data) {
		return -1
	}
	switch c := data[i]; {
	case c == '0':
		return i + 1
	case '1' <= c && c <= '9':
		return skipDigits(data, i+1)
	}
	return -1
}

func scanFraction(data []byte, i int) int {
	if i >= len(data) || data[i] != '.' {
		return i
	}
	j := skipDigits(data, i+1)
	if j == i+1 {
		return -1
	}
	return j
}

func scanExponent(data []byte, i int) int {
	if i >= len(data) || (data[i] != 'e' && data[i] != 'E') {
		return i
	}
	j := i + 1
	if j < len(data) && (data[j] == '+' || data[j] == '-') {
		j++
	}
	k := skipDigits(data, j)
	if k == j {
		return -1
	}
	return k
}

func scanNumber(data []byte, i int) int {
	if i < len(data) && data[i] == '-' {
		i++
	}
	if i = scanIntPart(data, i); i < 0 {
		return -1
	}
	if i = scanFraction(data, i); i < 0 {
		return -1
	}
	return scanExponent(data, i)
}

func scanWord(data []byte, i int, word string) int {
	if len(data)-i < len(word) || string(data[i:i+len(word)]) != word {
		return -1
	}
	return i + len(word)
}

func scanLiteral(data []byte, i int) int {
	switch data[i] {
	case 't':
		return scanWord(data, i, "true")
	case 'f':
		return scanWord(data, i, "false")
	}
	return scanWord(data, i, "null")
}

func scanValue(data []byte, i, depth int) int {
	i = skipSpace(data, i)
	if i >= len(data) {
		return -1
	}
	switch valueStart[data[i]] {
	case '"':
		end, _ := scanString(data, i)
		return end
	case '{':
		return scanObject(data, i, depth+1)
	case '[':
		return scanArray(data, i, depth+1)
	case '0':
		return scanNumber(data, i)
	case 't', 'f', 'n':
		return scanLiteral(data, i)
	}
	return -1
}

func scanArray(data []byte, i, depth int) int {
	if depth > maxDepth {
		return -1
	}
	i = skipSpace(data, i+1)
	if i < len(data) && data[i] == ']' {
		return i + 1
	}
	for {
		if i = scanValue(data, i, depth); i < 0 {
			return -1
		}
		if i = skipSpace(data, i); i >= len(data) {
			return -1
		}
		switch data[i] {
		case ',':
			i = skipSpace(data, i+1)
		case ']':
			return i + 1
		default:
			return -1
		}
	}
}

func scanMember(data []byte, i, depth int) int {
	if i >= len(data) || data[i] != '"' {
		return -1
	}
	end, _ := scanString(data, i)
	if end < 0 {
		return -1
	}
	if i = skipSpace(data, end); i >= len(data) || data[i] != ':' {
		return -1
	}
	return scanValue(data, i+1, depth)
}

func scanObject(data []byte, i, depth int) int {
	if depth > maxDepth {
		return -1
	}
	i = skipSpace(data, i+1)
	if i < len(data) && data[i] == '}' {
		return i + 1
	}
	for {
		if i = scanMember(data, i, depth); i < 0 {
			return -1
		}
		if i = skipSpace(data, i); i >= len(data) {
			return -1
		}
		switch data[i] {
		case ',':
			i = skipSpace(data, i+1)
		case '}':
			return i + 1
		default:
			return -1
		}
	}
}
