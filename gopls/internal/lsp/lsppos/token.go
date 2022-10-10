// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsppos

import (
	"errors"
	"go/ast"
	"go/token"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
)

// TokenMapper maps token.Pos to LSP positions for a single file.
type TokenMapper struct {
	// file is used for computing offsets.
	file *token.File

	// For now, just delegate to a Mapper for position calculation. As an
	// optimization we could avoid building the mapper and just use the file, but
	// then have to correctly adjust for newline-terminated files. It is easier
	// to just delegate unless performance becomes a concern.
	mapper *Mapper
}

// NewTokenMapper creates a new TokenMapper for the given content, using the
// provided file to compute offsets.
func NewTokenMapper(content []byte, file *token.File) *TokenMapper {
	return &TokenMapper{
		file:   file,
		mapper: NewMapper(content),
	}
}

// Position returns the protocol position corresponding to the given pos. It
// returns false if pos is out of bounds for the file being mapped.
func (m *TokenMapper) Position(pos token.Pos) (protocol.Position, bool) {
	offset, err := safetoken.Offset(m.file, pos)
	if err != nil {
		return protocol.Position{}, false
	}
	return m.mapper.Position(offset)
}

// Range returns the protocol range corresponding to the given start and end
// positions. It returns an error if start or end is out of bounds for the file
// being mapped.
func (m *TokenMapper) Range(start, end token.Pos) (protocol.Range, error) {
	startPos, ok := m.Position(start)
	if !ok {
		return protocol.Range{}, errors.New("invalid start position")
	}
	endPos, ok := m.Position(end)
	if !ok {
		return protocol.Range{}, errors.New("invalid end position")
	}

	return protocol.Range{Start: startPos, End: endPos}, nil
}

// NodeRange returns the protocol range corresponding to the span of the given
// node.
func (m *TokenMapper) NodeRange(n ast.Node) (protocol.Range, error) {
	return m.Range(n.Pos(), n.End())
}
