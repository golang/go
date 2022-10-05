// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/internal/diff"
)

// Pos represents a position in a text buffer.
// Both Line and Column are 0-indexed.
// Column counts runes.
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

// applyEdits applies the edits to a file with the specified lines,
// and returns a new slice containing the lines of the patched file.
// It is a wrapper around diff.Apply; see that function for preconditions.
func applyEdits(lines []string, edits []Edit) ([]string, error) {
	src := strings.Join(lines, "\n")

	// Build a table of byte offset of start of each line.
	lineOffset := make([]int, len(lines)+1)
	offset := 0
	for i, line := range lines {
		lineOffset[i] = offset
		offset += len(line) + len("\n")
	}
	lineOffset[len(lines)] = offset // EOF

	var badCol error
	posToOffset := func(pos Pos) int {
		offset := lineOffset[pos.Line]
		// Convert pos.Column (runes) to a UTF-8 byte offset.
		if pos.Line < len(lines) {
			for i := 0; i < pos.Column; i++ {
				r, sz := utf8.DecodeRuneInString(src[offset:])
				if r == '\n' && badCol == nil {
					badCol = fmt.Errorf("bad column")
				}
				offset += sz
			}
		}
		return offset
	}

	// Convert fake.Edits to diff.Edits
	diffEdits := make([]diff.Edit, len(edits))
	for i, edit := range edits {
		diffEdits[i] = diff.Edit{
			Start: posToOffset(edit.Start),
			End:   posToOffset(edit.End),
			New:   edit.Text,
		}
	}

	patched, err := diff.Apply(src, diffEdits)
	if err != nil {
		return nil, err
	}

	return strings.Split(patched, "\n"), badCol
}
