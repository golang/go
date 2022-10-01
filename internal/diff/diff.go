// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package diff computes differences between files or strings.
package diff

import (
	"sort"
	"strings"
)

// TODO(adonovan): switch to []byte throughout.
// But make clear that the operation is defined on runes, not bytes.
// Also:
// - delete LineEdits? (used only by Unified and test)
// - delete Lines (unused except by its test)

// An Edit describes the replacement of a portion of a file.
type Edit struct {
	Start, End int    // byte offsets of the region to replace
	New        string // the replacement
}

// SortEdits orders edits by their start offset.  The sort is stable
// so that edits with the same start offset will not be reordered.
func SortEdits(edits []Edit) {
	sort.SliceStable(edits, func(i int, j int) bool {
		return edits[i].Start < edits[j].Start
	})
}

// Apply applies a sequence of edits to the src buffer and
// returns the result.  It may panic or produce garbage if the edits
// are overlapping, out of bounds of src, or out of order.
//
// TODO(adonovan): this function must not panic if the edits aren't
// consistent with src, or with each other---especially when fed
// information from an untrusted source. It should probably be
// defensive against bad input and report an error in any of the above
// situations.
func Apply(src string, edits []Edit) string {
	SortEdits(edits) // TODO(adonovan): move to caller? What's the contract? Don't mutate arguments.

	var out strings.Builder
	// TODO(adonovan): opt: preallocate correct final size
	// by scanning the list of edits. (This can be done
	// in the same pass as detecting inconsistent edits.)
	last := 0
	for _, edit := range edits {
		start := edit.Start
		if start > last {
			out.WriteString(src[last:start])
			last = start
		}
		out.WriteString(edit.New)
		last = edit.End
	}
	if last < len(src) {
		out.WriteString(src[last:])
	}
	return out.String()
}

// LineEdits expands and merges a sequence of edits so that each
// resulting edit replaces one or more complete lines.
//
// It may panic or produce garbage if the edits
// are overlapping, out of bounds of src, or out of order.
// TODO(adonovan): see consistency note at Apply.
// We could hide this from the API so that we can enforce
// the precondition... but it seems like a reasonable feature.
func LineEdits(src string, edits []Edit) []Edit {
	SortEdits(edits) // TODO(adonovan): is this necessary? Move burden to caller?

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
	return edits // aligned

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
	return append(expanded, expandEdit(prev, src)) // flush final edit
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
	// (endCol is the zero-based column number of end.)
	end := edit.End
	if endCol := end - 1 - strings.LastIndex(src[:end], "\n"); endCol > 0 {
		if nl := strings.IndexByte(src[end:], '\n'); nl < 0 {
			edit.End = len(src) // extend to EOF
		} else {
			edit.End = end + nl + 1 // extend beyond \n
		}
		edit.New += src[end:edit.End]
	}

	return edit
}
