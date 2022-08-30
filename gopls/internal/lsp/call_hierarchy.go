// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
)

func (s *Server) prepareCallHierarchy(ctx context.Context, params *protocol.CallHierarchyPrepareParams) ([]protocol.CallHierarchyItem, error) {
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.Go)
	defer release()
	if !ok {
		return nil, err
	}

	return source.PrepareCallHierarchy(ctx, snapshot, fh, params.Position)
}

func (s *Server) incomingCalls(ctx context.Context, params *protocol.CallHierarchyIncomingCallsParams) ([]protocol.CallHierarchyIncomingCall, error) {
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.Item.URI, source.Go)
	defer release()
	if !ok {
		return nil, err
	}

	return source.IncomingCalls(ctx, snapshot, fh, params.Item.Range.Start)
}

func (s *Server) outgoingCalls(ctx context.Context, params *protocol.CallHierarchyOutgoingCallsParams) ([]protocol.CallHierarchyOutgoingCall, error) {
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.Item.URI, source.Go)
	defer release()
	if !ok {
		return nil, err
	}

	return source.OutgoingCalls(ctx, snapshot, fh, params.Item.Range.Start)
}
