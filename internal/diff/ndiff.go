// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff

import (
	"strings"
	"unicode/utf8"

	"golang.org/x/tools/internal/diff/lcs"
)

// maxDiffs is a limit on how deeply the lcs algorithm should search
// the value is just a guess
const maxDiffs = 30

// Strings computes the differences between two strings.
// (Both it and the diff in the myers package have type ComputeEdits, which
// is why the arguments are strings, not []bytes.)
// TODO(adonovan): opt: consider switching everything to []bytes, if
// that's the more common type in practice. Or provide both flavors?
func Strings(before, after string) []Edit {
	if before == after {
		// very frequently true
		return nil
	}
	// The diffs returned by the lcs package use indexes into
	// whatever slice was passed in. Edits use byte offsets, so
	// rune or line offsets need to be converted.
	// TODO(adonovan): opt: eliminate all the unnecessary allocations.
	var diffs []lcs.Diff
	if !isASCII(before) || !isASCII(after) {
		diffs, _ = lcs.Compute([]rune(before), []rune(after), maxDiffs/2)
		diffs = runeOffsets(diffs, []rune(before))
	} else {
		// Common case: pure ASCII. Avoid expansion to []rune slice.
		diffs, _ = lcs.Compute([]byte(before), []byte(after), maxDiffs/2)
	}
	return convertDiffs(diffs)
}

// Lines computes the differences between two list of lines.
// TODO(adonovan): unused except by its test. Do we actually need it?
func Lines(before, after []string) []Edit {
	diffs, _ := lcs.Compute(before, after, maxDiffs/2)
	diffs = lineOffsets(diffs, before)
	return convertDiffs(diffs)
	// the code is not coping with possible missing \ns at the ends
}

func convertDiffs(diffs []lcs.Diff) []Edit {
	ans := make([]Edit, len(diffs))
	for i, d := range diffs {
		ans[i] = Edit{d.Start, d.End, d.Text}
	}
	return ans
}

// convert diffs with rune offsets into diffs with byte offsets
func runeOffsets(diffs []lcs.Diff, src []rune) []lcs.Diff {
	var idx int
	var tmp strings.Builder // string because []byte([]rune) is illegal
	for i, d := range diffs {
		tmp.WriteString(string(src[idx:d.Start]))
		v := tmp.Len()
		tmp.WriteString(string(src[d.Start:d.End]))
		d.Start = v
		idx = d.End
		d.End = tmp.Len()
		diffs[i] = d
	}
	return diffs
}

// convert diffs with line offsets into diffs with byte offsets
func lineOffsets(diffs []lcs.Diff, src []string) []lcs.Diff {
	var idx int
	var tmp strings.Builder // bytes/
	for i, d := range diffs {
		tmp.WriteString(strJoin(src[idx:d.Start]))
		v := tmp.Len()
		tmp.WriteString(strJoin(src[d.Start:d.End]))
		d.Start = v
		idx = d.End
		d.End = tmp.Len()
		diffs[i] = d
	}
	return diffs
}

// join lines. (strings.Join doesn't add a trailing separator)
func strJoin(elems []string) string {
	if len(elems) == 0 {
		return ""
	}
	n := 0
	for i := 0; i < len(elems); i++ {
		n += len(elems[i])
	}

	var b strings.Builder
	b.Grow(n)
	for _, s := range elems {
		b.WriteString(s)
		//b.WriteByte('\n')
	}
	return b.String()
}

// isASCII reports whether s contains only ASCII.
func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}
