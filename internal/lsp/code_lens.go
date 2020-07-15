// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/mod"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func (s *Server) codeLens(ctx context.Context, params *protocol.CodeLensParams) ([]protocol.CodeLens, error) {
	snapshot, fh, ok, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	if !ok {
		return nil, err
	}
	switch fh.Kind() {
	case source.Mod:
		return mod.CodeLens(ctx, snapshot, fh.URI())
	case source.Go:
		return source.CodeLens(ctx, snapshot, fh)
	}
	// Unsupported file kind for a code action.
	return nil, nil
}
