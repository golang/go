// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (s *Server) rename(ctx context.Context, params *protocol.RenameParams) (*protocol.WorkspaceEdit, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	f, err := getGoFile(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	ident, err := source.Identifier(ctx, view, f, params.Position)
	if err != nil {
		return nil, err
	}
	edits, err := ident.Rename(ctx, view, params.NewName)
	if err != nil {
		return nil, err
	}
	changes := make(map[string][]protocol.TextEdit)
	for uri, textEdits := range edits {
		f, err := getGoFile(ctx, view, uri)
		if err != nil {
			return nil, err
		}
		m, err := getMapper(ctx, f)
		if err != nil {
			return nil, err
		}
		protocolEdits, err := source.ToProtocolEdits(m, textEdits)
		if err != nil {
			return nil, err
		}
		changes[string(uri)] = protocolEdits
	}

	return &protocol.WorkspaceEdit{Changes: &changes}, nil
}

func (s *Server) prepareRename(ctx context.Context, params *protocol.TextDocumentPositionParams) (*protocol.Range, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	f, err := getGoFile(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	m, err := getMapper(ctx, f)
	if err != nil {
		return nil, err
	}

	// Find the identifier at the position.
	ident, err := source.PrepareRename(ctx, view, f, params.Position)
	if err != nil {
		// Do not return the errors here, as it adds clutter.
		// Returning a nil result means there is not a valid rename.
		return nil, nil
	}
	identSpn, err := ident.Range.Span()
	if err != nil {
		return nil, err
	}

	identRng, err := m.Range(identSpn)
	if err != nil {
		return nil, err
	}
	// TODO(suzmue): return ident.Name as the placeholder text.
	return &identRng, nil
}
