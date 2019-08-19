// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package diff supports a pluggable diff algorithm.
package diff

import (
	"sort"

	"golang.org/x/tools/internal/span"
)

// TextEdit represents a change to a section of a document.
// The text within the specified span should be replaced by the supplied new text.
type TextEdit struct {
	Span    span.Span
	NewText string
}

var (
	ComputeEdits func(uri span.URI, before, after string) []TextEdit
	ApplyEdits   func(before string, edits []TextEdit) string
	ToUnified    func(from, to string, before string, edits []TextEdit) string
)

func SortTextEdits(d []TextEdit) {
	// Use a stable sort to maintain the order of edits inserted at the same position.
	sort.SliceStable(d, func(i int, j int) bool {
		return span.Compare(d[i].Span, d[j].Span) < 0
	})
}
