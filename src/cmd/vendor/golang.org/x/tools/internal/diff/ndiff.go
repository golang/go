// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff

import (
	"bytes"
	"unicode/utf8"

	"golang.org/x/tools/internal/diff/lcs"
)

// Strings computes the differences between two strings.
// The resulting edits respect rune boundaries.
func Strings(before, after string) []Edit {
	if before == after {
		return nil // common case
	}

	if isASCII(before) && isASCII(after) {
		// TODO(adonovan): opt: specialize diffASCII for strings.
		return diffASCII([]byte(before), []byte(after))
	}
	return diffRunes([]rune(before), []rune(after))
}

// Bytes computes the differences between two byte slices.
// The resulting edits respect rune boundaries.
func Bytes(before, after []byte) []Edit {
	if bytes.Equal(before, after) {
		return nil // common case
	}

	if isASCII(before) && isASCII(after) {
		return diffASCII(before, after)
	}
	return diffRunes(runes(before), runes(after))
}

func diffASCII(before, after []byte) []Edit {
	diffs := lcs.DiffBytes(before, after)

	// Convert from LCS diffs.
	res := make([]Edit, len(diffs))
	for i, d := range diffs {
		res[i] = Edit{d.Start, d.End, string(after[d.ReplStart:d.ReplEnd])}
	}
	return res
}

func diffRunes(before, after []rune) []Edit {
	diffs := lcs.DiffRunes(before, after)

	// The diffs returned by the lcs package use indexes
	// into whatever slice was passed in.
	// Convert rune offsets to byte offsets.
	res := make([]Edit, len(diffs))
	lastEnd := 0
	utf8Len := 0
	for i, d := range diffs {
		utf8Len += runesLen(before[lastEnd:d.Start]) // text between edits
		start := utf8Len
		utf8Len += runesLen(before[d.Start:d.End]) // text deleted by this edit
		res[i] = Edit{start, utf8Len, string(after[d.ReplStart:d.ReplEnd])}
		lastEnd = d.End
	}
	return res
}

// runes is like []rune(string(bytes)) without the duplicate allocation.
func runes(bytes []byte) []rune {
	n := utf8.RuneCount(bytes)
	runes := make([]rune, n)
	for i := range n {
		r, sz := utf8.DecodeRune(bytes)
		bytes = bytes[sz:]
		runes[i] = r
	}
	return runes
}

// runesLen returns the length in bytes of the UTF-8 encoding of runes.
func runesLen(runes []rune) (len int) {
	for _, r := range runes {
		len += utf8.RuneLen(r)
	}
	return len
}

// isASCII reports whether s contains only ASCII.
func isASCII[S string | []byte](s S) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}
