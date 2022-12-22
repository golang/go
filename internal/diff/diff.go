// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package diff computes differences between text files or strings.
package diff

import (
	"fmt"
	"sort"
	"strings"
)

// An Edit describes the replacement of a portion of a text file.
type Edit struct {
	Start, End int    // byte offsets of the region to replace
	New        string // the replacement
}

func (e Edit) String() string {
	return fmt.Sprintf("{Start:%d,End:%d,New:%s}", e.Start, e.End, e.New)
}

// Apply applies a sequence of edits to the src buffer and returns the
// result. Edits are applied in order of start offset; edits with the
// same start offset are applied in they order they were provided.
//
// Apply returns an error if any edit is out of bounds,
// or if any pair of edits is overlapping.
func Apply(src string, edits []Edit) (string, error) {
	edits, size, err := validate(src, edits)
	if err != nil {
		return "", err
	}

	// Apply edits.
	out := make([]byte, 0, size)
	lastEnd := 0
	for _, edit := range edits {
		if lastEnd < edit.Start {
			out = append(out, src[lastEnd:edit.Start]...)
		}
		out = append(out, edit.New...)
		lastEnd = edit.End
	}
	out = append(out, src[lastEnd:]...)

	if len(out) != size {
		panic("wrong size")
	}

	return string(out), nil
}

// validate checks that edits are consistent with src,
// and returns the size of the patched output.
// It may return a different slice.
func validate(src string, edits []Edit) ([]Edit, int, error) {
	if !sort.IsSorted(editsSort(edits)) {
		edits = append([]Edit(nil), edits...)
		SortEdits(edits)
	}

	// Check validity of edits and compute final size.
	size := len(src)
	lastEnd := 0
	for _, edit := range edits {
		if !(0 <= edit.Start && edit.Start <= edit.End && edit.End <= len(src)) {
			return nil, 0, fmt.Errorf("diff has out-of-bounds edits")
		}
		if edit.Start < lastEnd {
			return nil, 0, fmt.Errorf("diff has overlapping edits")
		}
		size += len(edit.New) + edit.Start - edit.End
		lastEnd = edit.End
	}

	return edits, size, nil
}

// SortEdits orders a slice of Edits by (start, end) offset.
// This ordering puts insertions (end = start) before deletions
// (end > start) at the same point, but uses a stable sort to preserve
// the order of multiple insertions at the same point.
// (Apply detects multiple deletions at the same point as an error.)
func SortEdits(edits []Edit) {
	sort.Stable(editsSort(edits))
}

type editsSort []Edit

func (a editsSort) Len() int { return len(a) }
func (a editsSort) Less(i, j int) bool {
	if cmp := a[i].Start - a[j].Start; cmp != 0 {
		return cmp < 0
	}
	return a[i].End < a[j].End
}
func (a editsSort) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

// lineEdits expands and merges a sequence of edits so that each
// resulting edit replaces one or more complete lines.
// See ApplyEdits for preconditions.
func lineEdits(src string, edits []Edit) ([]Edit, error) {
	edits, _, err := validate(src, edits)
	if err != nil {
		return nil, err
	}

	// Do all edits begin and end at the start of a line?
	// TODO(adonovan): opt: is this fast path necessary?
	// (Also, it complicates the result ownership.)
	for _, edit := range edits {
		if edit.Start >= len(src) || // insertion at EOF
			edit.Start > 0 && src[edit.Start-1] != '\n' || // not at line start
			edit.End > 0 && src[edit.End-1] != '\n' { // not at line start
			goto expand
		}
	}
	return edits, nil // aligned

expand:
	expanded := make([]Edit, 0, len(edits)) // a guess
	prev := edits[0]
	// TODO(adonovan): opt: start from the first misaligned edit.
	// TODO(adonovan): opt: avoid quadratic cost of string += string.
	for _, edit := range edits[1:] {
		between := src[prev.End:edit.Start]
		if !strings.Contains(between, "\n") {
			// overlapping lines: combine with previous edit.
			prev.New += between + edit.New
			prev.End = edit.End
		} else {
			// non-overlapping lines: flush previous edit.
			expanded = append(expanded, expandEdit(prev, src))
			prev = edit
		}
	}
	return append(expanded, expandEdit(prev, src)), nil // flush final edit
}

// expandEdit returns edit expanded to complete whole lines.
func expandEdit(edit Edit, src string) Edit {
	// Expand start left to start of line.
	// (delta is the zero-based column number of of start.)
	start := edit.Start
	if delta := start - 1 - strings.LastIndex(src[:start], "\n"); delta > 0 {
		edit.Start -= delta
		edit.New = src[start-delta:start] + edit.New
	}

	// Expand end right to end of line.
	end := edit.End
	if nl := strings.IndexByte(src[end:], '\n'); nl < 0 {
		edit.End = len(src) // extend to EOF
	} else {
		edit.End = end + nl + 1 // extend beyond \n
	}
	edit.New += src[end:edit.End]

	return edit
}
