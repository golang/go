// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/template"
)

func (s *Server) definition(ctx context.Context, params *protocol.DefinitionParams) ([]protocol.Location, error) {
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	defer release()
	if !ok {
		return nil, err
	}
	if snapshot.View().FileKind(fh) == source.Tmpl {
		return template.Definition(snapshot, fh, params.Position)
	}
	ident, err := source.Identifier(ctx, snapshot, fh, params.Position)
	if err != nil {
		return nil, err
	}
	if ident.IsImport() && !snapshot.View().Options().ImportShortcut.ShowDefinition() {
		return nil, nil
	}
	var locations []protocol.Location
	for _, ref := range ident.Declaration.MappedRange {
		locations = append(locations, ref.Location())
	}

	return locations, nil
}

func (s *Server) typeDefinition(ctx context.Context, params *protocol.TypeDefinitionParams) ([]protocol.Location, error) {
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.Go)
	defer release()
	if !ok {
		return nil, err
	}
	ident, err := source.Identifier(ctx, snapshot, fh, params.Position)
	if err != nil {
		return nil, err
	}
	if ident.Type.Object == nil {
		return nil, fmt.Errorf("no type definition for %s", ident.Name)
	}
	return []protocol.Location{ident.Type.MappedRange.Location()}, nil
}
