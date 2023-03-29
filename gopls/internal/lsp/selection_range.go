// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/event"
)

// selectionRange defines the textDocument/selectionRange feature,
// which, given a list of positions within a file,
// reports a linked list of enclosing syntactic blocks, innermost first.
//
// See https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_selectionRange.
//
// This feature can be used by a client to implement "expand selection" in a
// language-aware fashion. Multiple input positions are supported to allow
// for multiple cursors, and the entire path up to the whole document is
// returned for each cursor to avoid multiple round-trips when the user is
// likely to issue this command multiple times in quick succession.
func (s *Server) selectionRange(ctx context.Context, params *protocol.SelectionRangeParams) ([]protocol.SelectionRange, error) {
	ctx, done := event.Start(ctx, "lsp.Server.selectionRange")
	defer done()

	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	defer release()
	if !ok {
		return nil, err
	}

	pgf, err := snapshot.ParseGo(ctx, fh, source.ParseFull)
	if err != nil {
		return nil, err
	}

	result := make([]protocol.SelectionRange, len(params.Positions))
	for i, protocolPos := range params.Positions {
		pos, err := pgf.PositionPos(protocolPos)
		if err != nil {
			return nil, err
		}

		path, _ := astutil.PathEnclosingInterval(pgf.File, pos, pos)

		tail := &result[i] // tail of the Parent linked list, built head first

		for j, node := range path {
			rng, err := pgf.NodeRange(node)
			if err != nil {
				return nil, err
			}

			// Add node to tail.
			if j > 0 {
				tail.Parent = &protocol.SelectionRange{}
				tail = tail.Parent
			}
			tail.Range = rng
		}
	}

	return result, nil
}
