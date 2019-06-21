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
	f, m, err := getGoFile(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(params.Position)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	ident, err := source.Identifier(ctx, view, f, rng.Start)
	if err != nil {
		return nil, err
	}
	edits, err := ident.Rename(ctx, params.NewName)
	if err != nil {
		return nil, err
	}
	changes := make(map[string][]protocol.TextEdit)
	for uri, textEdits := range edits {
		_, m, err := getGoFile(ctx, view, uri)
		if err != nil {
			return nil, err
		}
		protocolEdits, err := ToProtocolEdits(m, textEdits)
		if err != nil {
			return nil, err
		}
		changes[string(uri)] = protocolEdits
	}

	return &protocol.WorkspaceEdit{Changes: &changes}, nil
}
