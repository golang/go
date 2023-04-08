// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"path/filepath"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
)

func (s *Server) rename(ctx context.Context, params *protocol.RenameParams) (*protocol.WorkspaceEdit, error) {
	ctx, done := event.Start(ctx, "lsp.Server.rename", tag.URI.Of(params.TextDocument.URI))
	defer done()

	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.Go)
	defer release()
	if !ok {
		return nil, err
	}
	// Because we don't handle directory renaming within source.Rename, source.Rename returns
	// boolean value isPkgRenaming to determine whether an DocumentChanges of type RenameFile should
	// be added to the return protocol.WorkspaceEdit value.
	edits, isPkgRenaming, err := source.Rename(ctx, snapshot, fh, params.Position, params.NewName)
	if err != nil {
		return nil, err
	}

	docChanges := []protocol.DocumentChanges{} // must be a slice
	for uri, e := range edits {
		fh, err := snapshot.ReadFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		docChanges = append(docChanges, documentChanges(fh, e)...)
	}
	if isPkgRenaming {
		// Update the last component of the file's enclosing directory.
		oldBase := filepath.Dir(fh.URI().Filename())
		newURI := filepath.Join(filepath.Dir(oldBase), params.NewName)
		docChanges = append(docChanges, protocol.DocumentChanges{
			RenameFile: &protocol.RenameFile{
				Kind:   "rename",
				OldURI: protocol.URIFromPath(oldBase),
				NewURI: protocol.URIFromPath(newURI),
			},
		})
	}
	return &protocol.WorkspaceEdit{
		DocumentChanges: docChanges,
	}, nil
}

// prepareRename implements the textDocument/prepareRename handler. It may
// return (nil, nil) if there is no rename at the cursor position, but it is
// not desirable to display an error to the user.
//
// TODO(rfindley): why wouldn't we want to show an error to the user, if the
// user initiated a rename request at the cursor?
func (s *Server) prepareRename(ctx context.Context, params *protocol.PrepareRenameParams) (*protocol.PrepareRename2Gn, error) {
	ctx, done := event.Start(ctx, "lsp.Server.prepareRename", tag.URI.Of(params.TextDocument.URI))
	defer done()

	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.Go)
	defer release()
	if !ok {
		return nil, err
	}
	// Do not return errors here, as it adds clutter.
	// Returning a nil result means there is not a valid rename.
	item, usererr, err := source.PrepareRename(ctx, snapshot, fh, params.Position)
	if err != nil {
		// Return usererr here rather than err, to avoid cluttering the UI with
		// internal error details.
		return nil, usererr
	}
	return &protocol.PrepareRename2Gn{
		Range:       item.Range,
		Placeholder: item.Text,
	}, nil
}
