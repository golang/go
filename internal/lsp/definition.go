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

func (s *Server) definition(ctx context.Context, params *protocol.DefinitionParams) ([]protocol.Location, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	ident, err := source.Identifier(ctx, view, f, params.Position)
	if err != nil {
		return nil, err
	}
	decRange, err := ident.Declaration.Range()
	if err != nil {
		return nil, err
	}
	return []protocol.Location{
		{
			URI:   protocol.NewURI(ident.Declaration.URI()),
			Range: decRange,
		},
	}, nil
}

func (s *Server) typeDefinition(ctx context.Context, params *protocol.TypeDefinitionParams) ([]protocol.Location, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	ident, err := source.Identifier(ctx, view, f, params.Position)
	if err != nil {
		return nil, err
	}
	identRange, err := ident.Type.Range()
	if err != nil {
		return nil, err
	}
	return []protocol.Location{
		{
			URI:   protocol.NewURI(ident.Type.URI()),
			Range: identRange,
		},
	}, nil
}
