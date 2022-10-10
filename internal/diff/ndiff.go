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

	if stringIsASCII(before) && stringIsASCII(after) {
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

	if bytesIsASCII(before) && bytesIsASCII(after) {
		return diffASCII(before, after)
	}
	return diffRunes(runes(before), runes(after))
}

func diffASCII(before, after []byte) []Edit {
	diffs, _ := lcs.Compute(before, after, maxDiffs/2)

	// Convert from LCS diffs.
	res := make([]Edit, len(diffs))
	for i, d := range diffs {
		res[i] = Edit{d.Start, d.End, d.Text}
	}
	return res
}

func diffRunes(before, after []rune) []Edit {
	diffs, _ := lcs.Compute(before, after, maxDiffs/2)

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
		res[i] = Edit{start, utf8Len, d.Text}
		lastEnd = d.End
	}
	return res
}

// maxDiffs is a limit on how deeply the lcs algorithm should search
// the value is just a guess
const maxDiffs = 30

// runes is like []rune(string(bytes)) without the duplicate allocation.
func runes(bytes []byte) []rune {
	n := utf8.RuneCount(bytes)
	runes := make([]rune, n)
	for i := 0; i < n; i++ {
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

// stringIsASCII reports whether s contains only ASCII.
// TODO(adonovan): combine when x/tools allows generics.
func stringIsASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

func bytesIsASCII(s []byte) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}
