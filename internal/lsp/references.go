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
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return nil, err
	}
	snapshot := view.Snapshot()
	fh, err := snapshot.GetFile(uri)
	if err != nil {
		return nil, err
	}
	// Find all references to the identifier at the position.
	if fh.Identity().Kind != source.Go {
		return nil, nil
	}

	references, err := source.References(ctx, view.Snapshot(), fh, params.Position, params.Context.IncludeDeclaration)
	if err != nil {
		return nil, err
	}

	var locations []protocol.Location
	for _, ref := range references {
		refRange, err := ref.Range()
		if err != nil {
			return nil, err
		}

		locations = append(locations, protocol.Location{
			URI:   protocol.NewURI(ref.URI()),
			Range: refRange,
		})
	}

	return locations, nil
}
