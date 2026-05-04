// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

// Package jsonwire implements stateless functionality for handling JSON text.
package jsonwire

import (
	"bytes"
	"cmp"
	"errors"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf16"
	"unicode/utf8"
)

// TrimSuffixWhitespace trims JSON from the end of b.
func TrimSuffixWhitespace(b []byte) []byte {
	// NOTE: The arguments and logic are kept simple to keep this inlinable.
	n := len(b) - 1
	for n >= 0 && (b[n] == ' ' || b[n] == '\t' || b[n] == '\r' || b[n] == '\n') {
		n--
	}
	return b[:n+1]
}

// TrimSuffixString trims a valid JSON string at the end of b.
// The behavior is undefined if there is not a valid JSON string present.
func TrimSuffixString(b []byte) []byte {
	// NOTE: The arguments and logic are kept simple to keep this inlinable.
	if len(b) > 0 && b[len(b)-1] == '"' {
		b = b[:len(b)-1]
	}
	for len(b) >= 2 && !(b[len(b)-1] == '"' && b[len(b)-2] != '\\') {
		b = b[:len(b)-1] // trim all characters except an unescaped quote
	}
	if len(b) > 0 && b[len(b)-1] == '"' {
		b = b[:len(b)-1]
	}
	return b
}

// HasSuffixByte reports whether b ends with c.
func HasSuffixByte(b []byte, c byte) bool {
	// NOTE: The arguments and logic are kept simple to keep this inlinable.
	return len(b) > 0 && b[len(b)-1] == c
}

// TrimSuffixByte removes c from the end of b if it is present.
func TrimSuffixByte(b []byte, c byte) []byte {
	// NOTE: The arguments and logic are kept simple to keep this inlinable.
	if len(b) > 0 && b[len(b)-1] == c {
		return b[:len(b)-1]
	}
	return b
}

// QuoteRune quotes the first rune in the input.
func QuoteRune(b []byte) string {
	r, n := utf8.DecodeRune(b)
	if r == utf8.RuneError && n == 1 {
		return `'\x` + strconv.FormatUint(uint64(b[0]), 16) + `'`
	}
	return strconv.QuoteRune(r)
}

// CompareUTF16 lexicographically compares x to y according
// to the UTF-16 codepoints of the UTF-8 encoded input strings.
// This implements the ordering specified in RFC 8785, section 3.2.3.
func CompareUTF16(x, y []byte) int {
	// NOTE: This is an optimized, mostly allocation-free implementation
	// of CompareUTF16Simple in wire_test.go. FuzzCompareUTF16 verifies that the
	// two implementations agree on the result of comparing any two strings.
	isUTF16Self := func(r rune) bool {
		return ('\u0000' <= r && r <= '\uD7FF') || ('\uE000' <= r && r <= '\uFFFF')
	}

	for {
		if len(x) == 0 || len(y) == 0 {
			return cmp.Compare(len(x), len(y))
		}

		// ASCII fast-path.
		if x[0] < utf8.RuneSelf || y[0] < utf8.RuneSelf {
			if x[0] != y[0] {
				return cmp.Compare(x[0], y[0])
			}
			x, y = x[1:], y[1:]
			continue
		}

		// Decode next pair of runes as UTF-8.
		rx, nx := utf8.DecodeRune(x)
		ry, ny := utf8.DecodeRune(y)

		selfx := isUTF16Self(rx)
		selfy := isUTF16Self(ry)
		switch {
		// The x rune is a single UTF-16 codepoint, while
		// the y rune is a surrogate pair of UTF-16 codepoints.
		case selfx && !selfy:
			ry, _ = utf16.EncodeRune(ry)
		// The y rune is a single UTF-16 codepoint, while
		// the x rune is a surrogate pair of UTF-16 codepoints.
		case selfy && !selfx:
			rx, _ = utf16.EncodeRune(rx)
		}
		if rx != ry {
			return cmp.Compare(rx, ry)
		}

		// Check for invalid UTF-8, in which case,
		// we just perform a byte-for-byte comparison.
		if isInvalidUTF8(rx, nx) || isInvalidUTF8(ry, ny) {
			if x[0] != y[0] {
				return cmp.Compare(x[0], y[0])
			}
		}
		x, y = x[nx:], y[ny:]
	}
}

// TODO(https://go.dev/issue/70547): Use utf8.ErrInvalid instead.
var ErrInvalidUTF8 = errors.New("invalid UTF-8")

func NewInvalidCharacterError[Bytes ~[]byte | ~string](prefix Bytes, where string) error {
	what := QuoteRune([]byte(prefix))
	return errors.New("invalid character " + what + " " + where)
}

func NewInvalidEscapeSequenceError[Bytes ~[]byte | ~string](what Bytes) error {
	label := "escape sequence"
	if len(what) > 6 {
		label = "surrogate pair"
	}
	needEscape := bytes.IndexFunc([]byte(what), func(r rune) bool {
		return r == '`' || r == utf8.RuneError || unicode.IsSpace(r) || !unicode.IsPrint(r)
	}) >= 0
	if needEscape {
		return errors.New("invalid " + label + " " + strconv.Quote(string(what)) + " in string")
	} else {
		return errors.New("invalid " + label + " `" + string(what) + "` in string")
	}
}

// TruncatePointer optionally truncates the JSON pointer,
// enforcing that the length roughly does not exceed n.
func TruncatePointer(s string, n int) string {
	if len(s) <= n {
		return s
	}
	i := n / 2
	j := len(s) - n/2

	// Avoid truncating a name if there are multiple names present.
	if k := strings.LastIndexByte(s[:i], '/'); k > 0 {
		i = k
	}
	if k := strings.IndexByte(s[j:], '/'); k >= 0 {
		j += k + len("/")
	}

	// Avoid truncation in the middle of a UTF-8 rune.
	for i > 0 && isInvalidUTF8(utf8.DecodeLastRuneInString(s[:i])) {
		i--
	}
	for j < len(s) && isInvalidUTF8(utf8.DecodeRuneInString(s[j:])) {
		j++
	}

	// Determine the right middle fragment to use.
	var middle string
	switch strings.Count(s[i:j], "/") {
	case 0:
		middle = "…"
	case 1:
		middle = "…/…"
	default:
		middle = "…/…/…"
	}
	if strings.HasPrefix(s[i:j], "/") && middle != "…" {
		middle = strings.TrimPrefix(middle, "…")
	}
	if strings.HasSuffix(s[i:j], "/") && middle != "…" {
		middle = strings.TrimSuffix(middle, "…")
	}
	return s[:i] + middle + s[j:]
}

func isInvalidUTF8(r rune, rn int) bool {
	return r == utf8.RuneError && rn == 1
}
