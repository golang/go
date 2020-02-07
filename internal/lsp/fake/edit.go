// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"fmt"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
)

// Pos represents a 0-indexed position in a text buffer.
type Pos struct {
	Line, Column int
}

func (p Pos) toProtocolPosition() protocol.Position {
	return protocol.Position{
		Line:      float64(p.Line),
		Character: float64(p.Column),
	}
}

func fromProtocolPosition(pos protocol.Position) Pos {
	return Pos{
		Line:   int(pos.Line),
		Column: int(pos.Character),
	}
}

// Edit represents a single (contiguous) buffer edit.
type Edit struct {
	Start, End Pos
	Text       string
}

// NewEdit creates an edit replacing all content between
// (startLine, startColumn) and (endLine, endColumn) with text.
func NewEdit(startLine, startColumn, endLine, endColumn int, text string) Edit {
	return Edit{
		Start: Pos{Line: startLine, Column: startColumn},
		End:   Pos{Line: endLine, Column: endColumn},
		Text:  text,
	}
}

func (e Edit) toProtocolChangeEvent() protocol.TextDocumentContentChangeEvent {
	return protocol.TextDocumentContentChangeEvent{
		Range: &protocol.Range{
			Start: e.Start.toProtocolPosition(),
			End:   e.End.toProtocolPosition(),
		},
		Text: e.Text,
	}
}

func fromProtocolTextEdit(textEdit protocol.TextEdit) Edit {
	return Edit{
		Start: fromProtocolPosition(textEdit.Range.Start),
		End:   fromProtocolPosition(textEdit.Range.End),
		Text:  textEdit.NewText,
	}
}

// inText reports whether p is a valid position in the text buffer.
func inText(p Pos, content []string) bool {
	if p.Line < 0 || p.Line >= len(content) {
		return false
	}
	// Note the strict right bound: the column indexes character _separators_,
	// not characters.
	if p.Column < 0 || p.Column > len(content[p.Line]) {
		return false
	}
	return true
}

// editContent implements a simplistic, inefficient algorithm for applying text
// edits to our buffer representation. It returns an error if the edit is
// invalid for the current content.
func editContent(content []string, edit Edit) ([]string, error) {
	if edit.End.Line < edit.Start.Line || (edit.End.Line == edit.Start.Line && edit.End.Column < edit.Start.Column) {
		return nil, fmt.Errorf("invalid edit: end %v before start %v", edit.End, edit.Start)
	}
	if !inText(edit.Start, content) {
		return nil, fmt.Errorf("start position %v is out of bounds", edit.Start)
	}
	if !inText(edit.End, content) {
		return nil, fmt.Errorf("end position %v is out of bounds", edit.End)
	}
	// Splice the edit text in between the first and last lines of the edit.
	prefix := string([]rune(content[edit.Start.Line])[:edit.Start.Column])
	suffix := string([]rune(content[edit.End.Line])[edit.End.Column:])
	newLines := strings.Split(prefix+edit.Text+suffix, "\n")
	newContent := append(content[:edit.Start.Line], newLines...)
	return append(newContent, content[edit.End.Line+1:]...), nil
}
