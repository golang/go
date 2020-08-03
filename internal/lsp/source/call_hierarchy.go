// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	errors "golang.org/x/xerrors"
)

// PrepareCallHierarchy returns an array of CallHierarchyItem for a file and the position within the file
func PrepareCallHierarchy(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]protocol.CallHierarchyItem, error) {
	ctx, done := event.Start(ctx, "source.prepareCallHierarchy")
	defer done()

	identifier, err := Identifier(ctx, snapshot, fh, pos)
	if err != nil {
		if errors.Is(err, ErrNoIdentFound) {
			event.Log(ctx, err.Error(), tag.Position.Of(pos))
		} else {
			event.Error(ctx, "error getting identifier", err, tag.Position.Of(pos))
		}
		return nil, nil
	}

	// if identifier is not of type function
	_, ok := identifier.Declaration.node.(*ast.FuncDecl)
	if !ok {
		event.Log(ctx, "invalid identifier type, expected funtion declaration", tag.Position.Of(pos))
		return nil, nil
	}
	rng, err := identifier.Range()
	if err != nil {
		return nil, err
	}
	callHierarchyItem := protocol.CallHierarchyItem{
		Name:           identifier.Name,
		Kind:           protocol.Function,
		Tags:           []protocol.SymbolTag{},
		Detail:         "func()",
		URI:            protocol.DocumentURI(fh.URI()),
		Range:          rng,
		SelectionRange: rng,
	}
	return []protocol.CallHierarchyItem{callHierarchyItem}, nil
}

// IncomingCalls returns an array of CallHierarchyIncomingCall for a file and the position within the file
func IncomingCalls(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]protocol.CallHierarchyIncomingCall, error) {
	ctx, done := event.Start(ctx, "source.incomingCalls")
	defer done()

	return []protocol.CallHierarchyIncomingCall{}, nil
}

// OutgoingCalls returns an array of CallHierarchyOutgoingCall for a file and the position within the file
func OutgoingCalls(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]protocol.CallHierarchyOutgoingCall, error) {
	ctx, done := event.Start(ctx, "source.outgoingCalls")
	defer done()

	return []protocol.CallHierarchyOutgoingCall{}, nil
}
