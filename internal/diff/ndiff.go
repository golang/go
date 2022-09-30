// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff

import (
	"go/token"
	"log"
	"strings"
	"unicode/utf8"

	"golang.org/x/tools/internal/diff/lcs"
	"golang.org/x/tools/internal/span"
)

// maxDiffs is a limit on how deeply the lcs algorithm should search
// the value is just a guess
const maxDiffs = 30

// Strings computes the differences between two strings.
// (Both it and the diff in the myers package have type ComputeEdits, which
// is why the arguments are strings, not []bytes.)
// TODO(adonovan): opt: consider switching everything to []bytes, if
// that's the more common type in practice. Or provide both flavors?
func Strings(uri span.URI, before, after string) []TextEdit {
	if before == after {
		// very frequently true
		return nil
	}
	// the diffs returned by the lcs package use indexes into whatever slice
	// was passed in. TextEdits need a span.Span which is computed with
	// byte offsets, so rune or line offsets need to be converted.
	// TODO(adonovan): opt: eliminate all the unnecessary allocations.
	if needrunes(before) || needrunes(after) {
		diffs, _ := lcs.Compute([]rune(before), []rune(after), maxDiffs/2)
		diffs = runeOffsets(diffs, []rune(before))
		return convertDiffs(uri, diffs, []byte(before))
	} else {
		diffs, _ := lcs.Compute([]byte(before), []byte(after), maxDiffs/2)
		return convertDiffs(uri, diffs, []byte(before))
	}
}

// Lines computes the differences between two list of lines.
// TODO(adonovan): unused except by its test. Do we actually need it?
func Lines(uri span.URI, before, after []string) []TextEdit {
	diffs, _ := lcs.Compute(before, after, maxDiffs/2)
	diffs = lineOffsets(diffs, before)
	return convertDiffs(uri, diffs, []byte(strJoin(before)))
	// the code is not coping with possible missing \ns at the ends
}

// convert diffs with byte offsets into diffs with line and column
func convertDiffs(uri span.URI, diffs []lcs.Diff, src []byte) []TextEdit {
	ans := make([]TextEdit, len(diffs))

	// Reuse the machinery of go/token to convert (content, offset) to (line, column).
	tf := token.NewFileSet().AddFile("", -1, len(src))
	tf.SetLinesForContent(src)

	offsetToPoint := func(offset int) span.Point {
		// Re-use span.ToPosition's EOF workaround.
		// It is infallible if the diffs are consistent with src.
		line, col, err := span.ToPosition(tf, offset)
		if err != nil {
			log.Fatalf("invalid offset: %v", err)
		}
		return span.NewPoint(line, col, offset)
	}

	for i, d := range diffs {
		start := offsetToPoint(d.Start)
		end := start
		if d.End != d.Start {
			end = offsetToPoint(d.End)
		}
		ans[i] = TextEdit{span.New(uri, start, end), d.Text}
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

// need runes is true if the string needs to be converted to []rune
// for random access
func needrunes(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return true
		}
	}
	return false
}
