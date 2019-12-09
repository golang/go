// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func (s *Server) definition(ctx context.Context, params *protocol.DefinitionParams) ([]protocol.Location, error) {
	snapshot, fh, ok, err := s.beginFileRequest(params.TextDocument.URI, source.Go)
	if !ok {
		return nil, err
	}
	ident, err := source.Identifier(ctx, snapshot, fh, params.Position)
	if err != nil {
		return nil, err
	}

	var locations []protocol.Location
	for _, ref := range ident.Declaration.MappedRange {
		decRange, err := ref.Range()
		if err != nil {
			return nil, err
		}

		locations = append(locations, protocol.Location{
			URI:   protocol.URIFromSpanURI(ref.URI()),
			Range: decRange,
		})
	}

	return locations, nil
}

func (s *Server) typeDefinition(ctx context.Context, params *protocol.TypeDefinitionParams) ([]protocol.Location, error) {
	snapshot, fh, ok, err := s.beginFileRequest(params.TextDocument.URI, source.Go)
	if !ok {
		return nil, err
	}
	ident, err := source.Identifier(ctx, snapshot, fh, params.Position)
	if err != nil {
		return nil, err
	}
	identRange, err := ident.Type.Range()
	if err != nil {
		return nil, err
	}
	return []protocol.Location{
		{
			URI:   protocol.URIFromSpanURI(ident.Type.URI()),
			Range: identRange,
		},
	}, nil
}
