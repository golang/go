// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package myers

import (
	"strings"

	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/span"
)

func ComputeEdits(uri span.URI, before, after string) []diff.TextEdit {
	u := SplitLines(before)
	f := SplitLines(after)
	return myersDiffToEdits(uri, Operations(u, f))
}

func myersDiffToEdits(uri span.URI, ops []*Op) []diff.TextEdit {
	edits := make([]diff.TextEdit, 0, len(ops))
	for _, op := range ops {
		s := span.New(uri, span.NewPoint(op.I1+1, 1, 0), span.NewPoint(op.I2+1, 1, 0))
		switch op.Kind {
		case diff.Delete:
			// Delete: unformatted[i1:i2] is deleted.
			edits = append(edits, diff.TextEdit{Span: s})
		case diff.Insert:
			// Insert: formatted[j1:j2] is inserted at unformatted[i1:i1].
			if content := strings.Join(op.Content, ""); content != "" {
				edits = append(edits, diff.TextEdit{Span: s, NewText: content})
			}
		}
	}
	return edits
}

func myersEditsToDiff(edits []diff.TextEdit) []*Op {
	iToJ := 0
	ops := make([]*Op, len(edits))
	for i, edit := range edits {
		i1 := edit.Span.Start().Line() - 1
		i2 := edit.Span.End().Line() - 1
		kind := diff.Insert
		if edit.NewText == "" {
			kind = diff.Delete
		}
		ops[i] = &Op{
			Kind:    kind,
			Content: SplitLines(edit.NewText),
			I1:      i1,
			I2:      i2,
			J1:      i1 + iToJ,
		}
		if kind == diff.Insert {
			iToJ += len(ops[i].Content)
		} else {
			iToJ -= i2 - i1
		}
	}
	return ops
}
