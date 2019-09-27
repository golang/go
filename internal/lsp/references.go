// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
)

func (s *Server) references(ctx context.Context, params *protocol.ReferenceParams) ([]protocol.Location, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	// Find all references to the identifier at the position.
	ident, err := source.Identifier(ctx, view, f, params.Position)
	if err != nil {
		return nil, err
	}
	references, err := ident.References(ctx)
	if err != nil {
		log.Error(ctx, "no references", err, tag.Of("Identifier", ident.Name))
	}

	// Get the location of each reference to return as the result.
	locations := make([]protocol.Location, 0, len(references))
	seen := make(map[span.Span]bool)
	for _, ref := range references {
		refSpan, err := ref.Span()
		if err != nil {
			return nil, err
		}
		if seen[refSpan] {
			continue // already added this location
		}
		seen[refSpan] = true
		refRange, err := ref.Range()
		if err != nil {
			return nil, err
		}
		locations = append(locations, protocol.Location{
			URI:   protocol.NewURI(ref.URI()),
			Range: refRange,
		})
	}
	// The declaration of this identifier may not be in the
	// scope that we search for references, so make sure
	// it is added to the beginning of the list if IncludeDeclaration
	// was specified.
	if params.Context.IncludeDeclaration {
		decSpan, err := ident.Declaration.Span()
		if err != nil {
			return nil, err
		}
		if !seen[decSpan] {
			rng, err := ident.Declaration.Range()
			if err != nil {
				return nil, err
			}
			locations = append([]protocol.Location{
				{
					URI:   protocol.NewURI(ident.Declaration.URI()),
					Range: rng,
				},
			}, locations...)
		}
	}
	return locations, nil
}
