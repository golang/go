// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmp"
	"fmt"
	"slices"
	"sort"

	"cmd/compile/internal/syntax"
)

// Edit replaces the source text from Start up to End with Text. Positions
// are the 1-based line and byte column of the original file; an edit with
// Start equal to End inserts Text.
type Edit struct {
	Start, End syntax.Pos
	Text       string
}

// splice is an Edit resolved to byte offsets of one source text.
type splice struct {
	start, end int
	text       string
}

// lineStarts returns the byte offset at which each line of src begins.
func lineStarts(src []byte) []int {
	starts := []int{0}
	for i, c := range src {
		if c == '\n' {
			starts = append(starts, i+1)
		}
	}
	return starts
}

// offsetOf converts a position to a byte offset into a text of size bytes.
// Columns are byte columns of the position's own line; a column past the
// line end is rejected even when its offset lands inside a later line.
func offsetOf(starts []int, size int, pos syntax.Pos) (int, error) {
	line, col := int(pos.Line()), int(pos.Col())
	if line < 1 || line > len(starts) || col < 1 {
		return 0, fmt.Errorf("position %d:%d is outside the source", line, col)
	}
	end := size
	if line < len(starts) {
		end = starts[line] - 1
	}
	off := starts[line-1] + col - 1
	if off > end {
		return 0, fmt.Errorf("position %d:%d is outside the source", line, col)
	}
	return off, nil
}

// resolve converts e to byte offsets and rejects an edit that belongs to
// another file or ends before it starts.
func resolve(src []byte, starts []int, base *syntax.PosBase, e Edit) (splice, error) {
	for _, pos := range []syntax.Pos{e.Start, e.End} {
		if pos.FileBase() != base {
			return splice{}, fmt.Errorf("edit position %v belongs to a different file than %s", pos, base.Filename())
		}
	}
	start, err := offsetOf(starts, len(src), e.Start)
	if err != nil {
		return splice{}, err
	}
	end, err := offsetOf(starts, len(src), e.End)
	if err != nil {
		return splice{}, err
	}
	if end < start {
		return splice{}, fmt.Errorf("edit at %v ends before it starts", e.Start)
	}
	return splice{start, end, e.Text}, nil
}

// ApplyEdits applies edits to src and returns the new text. base is the
// file base of the file whose positions the edits carry. Edits may be
// given in any order; they must not overlap. Insertions at one offset keep
// their input order.
func ApplyEdits(src []byte, base *syntax.PosBase, edits []Edit) ([]byte, error) {
	want := base.Pos().FileBase()
	starts := lineStarts(src)
	splices := make([]splice, len(edits))
	for i, e := range edits {
		s, err := resolve(src, starts, want, e)
		if err != nil {
			return nil, err
		}
		splices[i] = s
	}
	return applySplices(src, splices)
}

// sortSplices orders splices by start, then end, keeping the input order
// of equal splices.
func sortSplices(splices []splice) {
	slices.SortStableFunc(splices, func(a, b splice) int {
		return cmp.Or(cmp.Compare(a.start, b.start), cmp.Compare(a.end, b.end))
	})
}

// applySplices applies byte-offset splices to src. It sorts splices in
// place and reports an error when two of them overlap.
func applySplices(src []byte, splices []splice) ([]byte, error) {
	sortSplices(splices)
	out := make([]byte, 0, len(src))
	prev := 0
	for _, s := range splices {
		if s.start < prev {
			return nil, fmt.Errorf("overlapping edits at byte offset %d", s.start)
		}
		out = append(out, src[prev:s.start]...)
		out = append(out, s.text...)
		prev = s.end
	}
	return append(out, src[prev:]...), nil
}

// editsOf converts splices of one file to Edits positioned on base.
func editsOf(src []byte, base *syntax.PosBase, splices []splice) []Edit {
	starts := lineStarts(src)
	posAt := func(off int) syntax.Pos {
		line := sort.SearchInts(starts, off+1) - 1
		return syntax.MakePos(base, uint(line+1), uint(off-starts[line]+1))
	}
	sorted := slices.Clone(splices)
	sortSplices(sorted)
	edits := make([]Edit, len(sorted))
	for i, s := range sorted {
		edits[i] = Edit{Start: posAt(s.start), End: posAt(s.end), Text: s.text}
	}
	return edits
}
