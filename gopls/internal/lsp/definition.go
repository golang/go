// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"errors"
	"fmt"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/template"
)

func (s *Server) definition(ctx context.Context, params *protocol.DefinitionParams) ([]protocol.Location, error) {
	// TODO(rfindley): definition requests should be multiplexed across all views.
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	defer release()
	if !ok {
		return nil, err
	}
	switch kind := snapshot.View().FileKind(fh); kind {
	case source.Tmpl:
		return template.Definition(snapshot, fh, params.Position)
	case source.Go:
		// Partial support for jumping from linkname directive (position at 2nd argument).
		locations, err := source.LinknameDefinition(ctx, snapshot, fh, params.Position)
		if !errors.Is(err, source.ErrNoLinkname) {
			return locations, err
		}
		return source.Definition(ctx, snapshot, fh, params.Position)
	default:
		return nil, fmt.Errorf("can't find definitions for file type %s", kind)
	}
}

func (s *Server) typeDefinition(ctx context.Context, params *protocol.TypeDefinitionParams) ([]protocol.Location, error) {
	// TODO(rfindley): type definition requests should be multiplexed across all views.
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.Go)
	defer release()
	if !ok {
		return nil, err
	}
	switch kind := snapshot.View().FileKind(fh); kind {
	case source.Go:
		return source.TypeDefinition(ctx, snapshot, fh, params.Position)
	default:
		return nil, fmt.Errorf("can't find type definitions for file type %s", kind)
	}
}
