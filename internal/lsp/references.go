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

func (s *Server) references(ctx context.Context, params *protocol.ReferenceParams) ([]protocol.Location, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	f, m, err := getGoFile(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(params.Position)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	// Find all references to the identifier at the position.
	ident, err := source.Identifier(ctx, view, f, rng.Start)
	if err != nil {
		return nil, err
	}
	references, err := ident.References(ctx)
	if err != nil {
		view.Session().Logger().Errorf(ctx, "no references for %s: %v", ident.Name, err)
	}
	if params.Context.IncludeDeclaration {
		// The declaration of this identifier may not be in the
		// scope that we search for references, so make sure
		// it is added to the beginning of the list if IncludeDeclaration
		// was specified.
		references = append([]*source.ReferenceInfo{
			&source.ReferenceInfo{
				Range: ident.DeclarationRange(),
			},
		}, references...)
	}

	// Get the location of each reference to return as the result.
	locations := make([]protocol.Location, 0, len(references))
	seen := make(map[span.Span]bool)
	for _, ref := range references {
		refSpan, err := ref.Range.Span()
		if err != nil {
			return nil, err
		}
		if seen[refSpan] {
			continue // already added this location
		}
		seen[refSpan] = true

		_, refM, err := getSourceFile(ctx, view, refSpan.URI())
		if err != nil {
			return nil, err
		}
		loc, err := refM.Location(refSpan)
		if err != nil {
			return nil, err
		}
		locations = append(locations, loc)
	}
	return locations, nil
}
