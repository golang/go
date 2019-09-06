// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff

import (
	"fmt"
	"strings"

	"golang.org/x/tools/internal/lsp/diff/myers"
	"golang.org/x/tools/internal/span"
)

func init() {
	ComputeEdits = myersComputeEdits
	ToUnified = myersToUnified
}

func myersComputeEdits(uri span.URI, before, after string) []TextEdit {
	u := myers.SplitLines(before)
	f := myers.SplitLines(after)
	return myersDiffToEdits(uri, myers.Operations(u, f))
}

func myersToUnified(from, to string, before string, edits []TextEdit) string {
	u := myers.SplitLines(before)
	ops := myersEditsToDiff(edits)
	return fmt.Sprint(myers.ToUnified(from, to, u, ops))
}

func myersDiffToEdits(uri span.URI, ops []*myers.Op) []TextEdit {
	edits := make([]TextEdit, 0, len(ops))
	for _, op := range ops {
		s := span.New(uri, span.NewPoint(op.I1+1, 1, 0), span.NewPoint(op.I2+1, 1, 0))
		switch op.Kind {
		case myers.Delete:
			// Delete: unformatted[i1:i2] is deleted.
			edits = append(edits, TextEdit{Span: s})
		case myers.Insert:
			// Insert: formatted[j1:j2] is inserted at unformatted[i1:i1].
			if content := strings.Join(op.Content, ""); content != "" {
				edits = append(edits, TextEdit{Span: s, NewText: content})
			}
		}
	}
	return edits
}

func myersEditsToDiff(edits []TextEdit) []*myers.Op {
	iToJ := 0
	ops := make([]*myers.Op, len(edits))
	for i, edit := range edits {
		i1 := edit.Span.Start().Line() - 1
		i2 := edit.Span.End().Line() - 1
		kind := myers.Insert
		if edit.NewText == "" {
			kind = myers.Delete
		}
		ops[i] = &myers.Op{
			Kind:    kind,
			Content: myers.SplitLines(edit.NewText),
			I1:      i1,
			I2:      i2,
			J1:      i1 + iToJ,
		}
		if kind == myers.Insert {
			iToJ += len(ops[i].Content)
		} else {
			iToJ -= i2 - i1
		}
	}
	return ops
}
