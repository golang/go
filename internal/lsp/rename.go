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
	fh, err := snapshot.GetFile(uri)
	if err != nil {
		return nil, err
	}
	if fh.Identity().Kind != source.Go {
		return nil, nil
	}

	edits, err := source.Rename(ctx, snapshot, fh, params.Position, params.NewName)
	if err != nil {
		return nil, err
	}

	var docChanges []protocol.TextDocumentEdit
	for uri, e := range edits {
		fh, err := snapshot.GetFile(uri)
		if err != nil {
			return nil, err
		}
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
	fh, err := snapshot.GetFile(uri)
	if err != nil {
		return nil, err
	}
	if fh.Identity().Kind != source.Go {
		return nil, nil
	}

	// Do not return errors here, as it adds clutter.
	// Returning a nil result means there is not a valid rename.
	item, err := source.PrepareRename(ctx, snapshot, fh, params.Position)
	if err != nil {
		return nil, nil // ignore errors
	}
	// TODO(suzmue): return ident.Name as the placeholder text.
	return &item.Range, nil
}
