// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/internal/diff"
)

// NewEdit creates an edit replacing all content between the 0-based
// (startLine, startColumn) and (endLine, endColumn) with text.
//
// Columns measure UTF-16 codes.
func NewEdit(startLine, startColumn, endLine, endColumn uint32, text string) protocol.TextEdit {
	return protocol.TextEdit{
		Range: protocol.Range{
			Start: protocol.Position{Line: startLine, Character: startColumn},
			End:   protocol.Position{Line: endLine, Character: endColumn},
		},
		NewText: text,
	}
}

func EditToChangeEvent(e protocol.TextEdit) protocol.TextDocumentContentChangeEvent {
	var rng protocol.Range = e.Range
	return protocol.TextDocumentContentChangeEvent{
		Range: &rng,
		Text:  e.NewText,
	}
}

// applyEdits applies the edits to a file with the specified lines,
// and returns a new slice containing the lines of the patched file.
// It is a wrapper around diff.Apply; see that function for preconditions.
func applyEdits(mapper *protocol.Mapper, edits []protocol.TextEdit, windowsLineEndings bool) ([]byte, error) {
	// Convert fake.Edits to diff.Edits
	diffEdits := make([]diff.Edit, len(edits))
	for i, edit := range edits {
		start, end, err := mapper.RangeOffsets(edit.Range)
		if err != nil {
			return nil, err
		}
		diffEdits[i] = diff.Edit{
			Start: start,
			End:   end,
			New:   edit.NewText,
		}
	}

	patchedString, err := diff.Apply(string(mapper.Content), diffEdits)
	if err != nil {
		return nil, err
	}
	patched := []byte(patchedString)
	if windowsLineEndings {
		patched = toWindowsLineEndings(patched)
	}
	return patched, nil
}
