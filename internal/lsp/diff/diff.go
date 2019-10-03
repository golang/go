// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package diff supports a pluggable diff algorithm.
package diff

import (
	"sort"
	"strings"

	"golang.org/x/tools/internal/span"
)

// TextEdit represents a change to a section of a document.
// The text within the specified span should be replaced by the supplied new text.
type TextEdit struct {
	Span    span.Span
	NewText string
}

// ComputeEdits is the type for a function that produces a set of edits that
// convert from the before content to the after content.
type ComputeEdits func(uri span.URI, before, after string) []TextEdit

// SortTextEdits attempts to order all edits by their starting points.
// The sort is stable so that edits with the same starting point will not
// be reordered.
func SortTextEdits(d []TextEdit) {
	// Use a stable sort to maintain the order of edits inserted at the same position.
	sort.SliceStable(d, func(i int, j int) bool {
		return span.Compare(d[i].Span, d[j].Span) < 0
	})
}

// ApplyEdits applies the set of edits to the before and returns the resulting
// content.
// It may panic or produce garbage if the edits are not valid for the provided
// before content.
func ApplyEdits(before string, edits []TextEdit) string {
	// Preconditions:
	//   - all of the edits apply to before
	//   - and all the spans for each TextEdit have the same URI
	if len(edits) == 0 {
		return before
	}
	edits = prepareEdits(edits)
	c := span.NewContentConverter("", []byte(before))
	after := strings.Builder{}
	last := 0
	for _, edit := range edits {
		spn, _ := edit.Span.WithAll(c)
		start := spn.Start().Offset()
		if start > last {
			after.WriteString(before[last:start])
			last = start
		}
		after.WriteString(edit.NewText)
		last = spn.End().Offset()
	}
	if last < len(before) {
		after.WriteString(before[last:])
	}
	return after.String()
}

// prepareEdits returns a sorted copy of the edits
func prepareEdits(edits []TextEdit) []TextEdit {
	copied := make([]TextEdit, len(edits))
	copy(copied, edits)
	SortTextEdits(copied)
	return copied
}
