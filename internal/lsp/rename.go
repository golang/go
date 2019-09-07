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
	for uri, e := range edits {
		changes[protocol.NewURI(uri)] = e
	}

	return &protocol.WorkspaceEdit{Changes: &changes}, nil
}

func (s *Server) prepareRename(ctx context.Context, params *protocol.PrepareRenameParams) (*protocol.Range, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	f, err := getGoFile(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	// Do not return errors here, as it adds clutter.
	// Returning a nil result means there is not a valid rename.
	item, err := source.PrepareRename(ctx, view, f, params.Position)
	if err != nil {
		return nil, nil
	}
	// TODO(suzmue): return ident.Name as the placeholder text.
	return &item.Range, nil
}
