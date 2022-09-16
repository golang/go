// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"fmt"
	"sort"
	"strings"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

// Pos represents a position in a text buffer. Both Line and Column are
// 0-indexed.
type Pos struct {
	Line, Column int
}

func (p Pos) String() string {
	return fmt.Sprintf("%v:%v", p.Line, p.Column)
}

// Range corresponds to protocol.Range, but uses the editor friend Pos
// instead of UTF-16 oriented protocol.Position
type Range struct {
	Start Pos
	End   Pos
}

func (p Pos) ToProtocolPosition() protocol.Position {
	return protocol.Position{
		Line:      uint32(p.Line),
		Character: uint32(p.Column),
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

// Location is the editor friendly equivalent of protocol.Location
type Location struct {
	Path  string
	Range Range
}

// SymbolInformation is an editor friendly version of
// protocol.SymbolInformation, with location information transformed to byte
// offsets. Field names correspond to the protocol type.
type SymbolInformation struct {
	Name     string
	Kind     protocol.SymbolKind
	Location Location
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
			Start: e.Start.ToProtocolPosition(),
			End:   e.End.ToProtocolPosition(),
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
	if p.Column < 0 || p.Column > len([]rune(content[p.Line])) {
		return false
	}
	return true
}

// editContent implements a simplistic, inefficient algorithm for applying text
// edits to our buffer representation. It returns an error if the edit is
// invalid for the current content.
//
// TODO(rfindley): this function does not handle non-ascii text correctly.
// TODO(rfindley): replace this with diff.ApplyEdits: we should not be
// maintaining an additional representation of edits.
func editContent(content []string, edits []Edit) ([]string, error) {
	newEdits := make([]Edit, len(edits))
	copy(newEdits, edits)
	sort.SliceStable(newEdits, func(i, j int) bool {
		ei := newEdits[i]
		ej := newEdits[j]

		// Sort by edit start position followed by end position. Given an edit
		// 3:1-3:1 followed by an edit 3:1-3:15, we must process the empty edit
		// first.
		if cmp := comparePos(ei.Start, ej.Start); cmp != 0 {
			return cmp < 0
		}

		return comparePos(ei.End, ej.End) < 0
	})

	// Validate edits.
	for _, edit := range newEdits {
		if edit.End.Line < edit.Start.Line || (edit.End.Line == edit.Start.Line && edit.End.Column < edit.Start.Column) {
			return nil, fmt.Errorf("invalid edit: end %v before start %v", edit.End, edit.Start)
		}
		if !inText(edit.Start, content) {
			return nil, fmt.Errorf("start position %v is out of bounds", edit.Start)
		}
		if !inText(edit.End, content) {
			return nil, fmt.Errorf("end position %v is out of bounds", edit.End)
		}
	}

	var (
		b            strings.Builder
		line, column int
	)
	advance := func(toLine, toColumn int) {
		for ; line < toLine; line++ {
			b.WriteString(string([]rune(content[line])[column:]) + "\n")
			column = 0
		}
		b.WriteString(string([]rune(content[line])[column:toColumn]))
		column = toColumn
	}
	for _, edit := range newEdits {
		advance(edit.Start.Line, edit.Start.Column)
		b.WriteString(edit.Text)
		line = edit.End.Line
		column = edit.End.Column
	}
	advance(len(content)-1, len([]rune(content[len(content)-1])))
	return strings.Split(b.String(), "\n"), nil
}

// comparePos returns -1 if left < right, 0 if left == right, and 1 if left > right.
func comparePos(left, right Pos) int {
	if left.Line < right.Line {
		return -1
	}
	if left.Line > right.Line {
		return 1
	}
	if left.Column < right.Column {
		return -1
	}
	if left.Column > right.Column {
		return 1
	}
	return 0
}
