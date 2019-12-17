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
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return nil, err
	}
	snapshot := view.Snapshot()
	fh, err := snapshot.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	// Find all references to the identifier at the position.
	if fh.Identity().Kind != source.Go {
		return nil, nil
	}
	phs, err := snapshot.PackageHandles(ctx, fh)
	if err != nil {
		return nil, nil
	}

	// Get the location of each reference to return as the result.
	var (
		locations []protocol.Location
		seen      = make(map[span.Span]bool)
		lastIdent *source.IdentifierInfo
	)
	for _, ph := range phs {
		ident, err := source.Identifier(ctx, snapshot, fh, params.Position, source.SpecificPackageHandle(ph.ID()))
		if err != nil {
			if err == source.ErrNoIdentFound {
				return nil, err
			}
			log.Error(ctx, "no identifier", err, tag.Of("Identifier", ident.Name))
			continue
		}

		lastIdent = ident

		references, err := ident.References(ctx)
		if err != nil {
			log.Error(ctx, "no references", err, tag.Of("Identifier", ident.Name))
			continue
		}

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
	}

	// Only add the identifier's declaration if the client requests it.
	if params.Context.IncludeDeclaration && lastIdent != nil {
		rng, err := lastIdent.Declaration.Range()
		if err != nil {
			return nil, err
		}
		locations = append([]protocol.Location{
			{
				URI:   protocol.NewURI(lastIdent.Declaration.URI()),
				Range: rng,
			},
		}, locations...)
	}

	return locations, nil
}
