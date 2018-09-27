package lsp

import (
	"fmt"
	"go/format"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
)

// format formats a document with a given range.
func (s *server) format(uri protocol.DocumentURI, rng *protocol.Range) ([]protocol.TextEdit, error) {
	data, err := s.readActiveFile(uri)
	if err != nil {
		return nil, err
	}
	if rng != nil {
		start, err := positionToOffset(data, int(rng.Start.Line), int(rng.Start.Character))
		if err != nil {
			return nil, err
		}
		end, err := positionToOffset(data, int(rng.End.Line), int(rng.End.Character))
		if err != nil {
			return nil, err
		}
		data = data[start:end]
		// format.Source will fail if the substring is not a balanced expression tree.
		// TODO(rstambler): parse the file and use astutil.PathEnclosingInterval to
		// find the largest ast.Node n contained within start:end, and format the
		// region n.Pos-n.End instead.
	}
	// format.Source changes slightly from one release to another, so the version
	// of Go used to build the LSP server will determine how it formats code.
	// This should be acceptable for all users, who likely be prompted to rebuild
	// the LSP server on each Go release.
	fmted, err := format.Source([]byte(data))
	if err != nil {
		return nil, err
	}
	if rng == nil {
		// Get the ending line and column numbers for the original file.
		line := strings.Count(data, "\n")
		col := len(data) - strings.LastIndex(data, "\n") - 1
		if col < 0 {
			col = 0
		}
		rng = &protocol.Range{
			Start: protocol.Position{0, 0},
			End:   protocol.Position{float64(line), float64(col)},
		}
	}
	// TODO(rstambler): Compute text edits instead of replacing whole file.
	return []protocol.TextEdit{
		{
			Range:   *rng,
			NewText: string(fmted),
		},
	}, nil
}

// positionToOffset converts a 0-based line and column number in a file
// to a byte offset value.
func positionToOffset(contents string, line, col int) (int, error) {
	start := 0
	for i := 0; i < int(line); i++ {
		if start >= len(contents) {
			return 0, fmt.Errorf("file contains %v lines, not %v lines", i, line)
		}
		index := strings.IndexByte(contents[start:], '\n')
		if index == -1 {
			return 0, fmt.Errorf("file contains %v lines, not %v lines", i, line)
		}
		start += (index + 1)
	}
	offset := start + int(col)
	return offset, nil
}
