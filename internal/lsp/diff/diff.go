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
type ComputeEdits func(uri span.URI, before, after string) ([]TextEdit, error)

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
	edits, _ = prepareEdits(before, edits)
	after := strings.Builder{}
	last := 0
	for _, edit := range edits {
		start := edit.Span.Start().Offset()
		if start > last {
			after.WriteString(before[last:start])
			last = start
		}
		after.WriteString(edit.NewText)
		last = edit.Span.End().Offset()
	}
	if last < len(before) {
		after.WriteString(before[last:])
	}
	return after.String()
}

// LineEdits takes a set of edits and expands and merges them as necessary
// to ensure that there are only full line edits left when it is done.
func LineEdits(before string, edits []TextEdit) []TextEdit {
	if len(edits) == 0 {
		return nil
	}
	edits, partial := prepareEdits(before, edits)
	if partial {
		edits = lineEdits(before, edits)
	}
	return edits
}

// prepareEdits returns a sorted copy of the edits
func prepareEdits(before string, edits []TextEdit) ([]TextEdit, bool) {
	partial := false
	tf := span.NewTokenFile("", []byte(before))
	copied := make([]TextEdit, len(edits))
	for i, edit := range edits {
		edit.Span, _ = edit.Span.WithAll(tf)
		copied[i] = edit
		partial = partial ||
			edit.Span.Start().Offset() >= len(before) ||
			edit.Span.Start().Column() > 1 || edit.Span.End().Column() > 1
	}
	SortTextEdits(copied)
	return copied, partial
}

// lineEdits rewrites the edits to always be full line edits
func lineEdits(before string, edits []TextEdit) []TextEdit {
	adjusted := make([]TextEdit, 0, len(edits))
	current := TextEdit{Span: span.Invalid}
	for _, edit := range edits {
		if current.Span.IsValid() && edit.Span.Start().Line() <= current.Span.End().Line() {
			// overlaps with the current edit, need to combine
			// first get the gap from the previous edit
			gap := before[current.Span.End().Offset():edit.Span.Start().Offset()]
			// now add the text of this edit
			current.NewText += gap + edit.NewText
			// and then adjust the end position
			current.Span = span.New(current.Span.URI(), current.Span.Start(), edit.Span.End())
		} else {
			// does not overlap, add previous run (if there is one)
			adjusted = addEdit(before, adjusted, current)
			// and then remember this edit as the start of the next run
			current = edit
		}
	}
	// add the current pending run if there is one
	return addEdit(before, adjusted, current)
}

func addEdit(before string, edits []TextEdit, edit TextEdit) []TextEdit {
	if !edit.Span.IsValid() {
		return edits
	}
	// if edit is partial, expand it to full line now
	start := edit.Span.Start()
	end := edit.Span.End()
	if start.Column() > 1 {
		// prepend the text and adjust to start of line
		delta := start.Column() - 1
		start = span.NewPoint(start.Line(), 1, start.Offset()-delta)
		edit.Span = span.New(edit.Span.URI(), start, end)
		edit.NewText = before[start.Offset():start.Offset()+delta] + edit.NewText
	}
	if start.Offset() >= len(before) && start.Line() > 1 && before[len(before)-1] != '\n' {
		// after end of file that does not end in eol, so join to last line of file
		// to do this we need to know where the start of the last line was
		eol := strings.LastIndex(before, "\n")
		if eol < 0 {
			// file is one non terminated line
			eol = 0
		}
		delta := len(before) - eol
		start = span.NewPoint(start.Line()-1, 1, start.Offset()-delta)
		edit.Span = span.New(edit.Span.URI(), start, end)
		edit.NewText = before[start.Offset():start.Offset()+delta] + edit.NewText
	}
	if end.Column() > 1 {
		remains := before[end.Offset():]
		eol := strings.IndexRune(remains, '\n')
		if eol < 0 {
			eol = len(remains)
		} else {
			eol++
		}
		end = span.NewPoint(end.Line()+1, 1, end.Offset()+eol)
		edit.Span = span.New(edit.Span.URI(), start, end)
		edit.NewText = edit.NewText + remains[:eol]
	}
	edits = append(edits, edit)
	return edits
}
