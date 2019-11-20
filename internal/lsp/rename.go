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
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return nil, err
	}
	snapshot := view.Snapshot()
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	ident, err := source.Identifier(ctx, snapshot, f, params.Position)
	if err != nil {
		return nil, err
	}
	edits, err := ident.Rename(ctx, params.NewName)
	if err != nil {
		return nil, err
	}
	var docChanges []protocol.TextDocumentEdit
	for uri, e := range edits {
		f, err := view.GetFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		fh := ident.Snapshot.Handle(ctx, f)
		docChanges = append(docChanges, documentChanges(fh, e)...)
	}
	return &protocol.WorkspaceEdit{
		DocumentChanges: docChanges,
	}, nil
}

func (s *Server) prepareRename(ctx context.Context, params *protocol.PrepareRenameParams) (*protocol.Range, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return nil, err
	}
	snapshot := view.Snapshot()
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	ident, err := source.Identifier(ctx, snapshot, f, params.Position)
	if err != nil {
		return nil, nil // ignore errors
	}
	// Do not return errors here, as it adds clutter.
	// Returning a nil result means there is not a valid rename.
	item, err := ident.PrepareRename(ctx)
	if err != nil {
		return nil, nil // ignore errors
	}
	// TODO(suzmue): return ident.Name as the placeholder text.
	return &item.Range, nil
}
