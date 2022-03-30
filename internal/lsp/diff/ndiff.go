// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff

import (
	"strings"
	"unicode/utf8"

	"golang.org/x/tools/internal/lsp/diff/lcs"
	"golang.org/x/tools/internal/span"
)

// maxDiffs is a limit on how deeply the lcs algorithm should search
// the value is just a guess
const maxDiffs = 30

// NComputeEdits computes TextEdits for strings
// (both it and the diff in the myers package have type ComputeEdits, which
// is why the arguments are strings, not []bytes.)
func NComputeEdits(uri span.URI, before, after string) ([]TextEdit, error) {
	if before == after {
		// very frequently true
		return nil, nil
	}
	// the diffs returned by the lcs package use indexes into whatever slice
	// was passed in. TextEdits  need a span.Span which is computed with
	// byte offsets, so rune or line offsets need to be converted.
	if needrunes(before) || needrunes(after) {
		diffs, _ := lcs.Compute([]rune(before), []rune(after), maxDiffs/2)
		diffs = runeOffsets(diffs, []rune(before))
		ans, err := convertDiffs(uri, diffs, []byte(before))
		return ans, err
	} else {
		diffs, _ := lcs.Compute([]byte(before), []byte(after), maxDiffs/2)
		ans, err := convertDiffs(uri, diffs, []byte(before))
		return ans, err
	}
}

// NComputeLineEdits computes TextEdits for []strings
func NComputeLineEdits(uri span.URI, before, after []string) ([]TextEdit, error) {
	diffs, _ := lcs.Compute(before, after, maxDiffs/2)
	diffs = lineOffsets(diffs, before)
	ans, err := convertDiffs(uri, diffs, []byte(strJoin(before)))
	// the code is not coping with possible missing \ns at the ends
	return ans, err
}

// convert diffs with byte offsets into diffs with line and column
func convertDiffs(uri span.URI, diffs []lcs.Diff, src []byte) ([]TextEdit, error) {
	ans := make([]TextEdit, len(diffs))
	tf := span.NewTokenFile(uri.Filename(), src)
	for i, d := range diffs {
		s := newSpan(uri, d.Start, d.End)
		s, err := s.WithPosition(tf)
		if err != nil {
			return nil, err
		}
		ans[i] = TextEdit{s, d.Text}
	}
	return ans, nil
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

func newSpan(uri span.URI, left, right int) span.Span {
	return span.New(uri, span.NewPoint(0, 0, left), span.NewPoint(0, 0, right))
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
