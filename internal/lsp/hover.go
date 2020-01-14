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

func (s *Server) hover(ctx context.Context, params *protocol.HoverParams) (*protocol.Hover, error) {
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
	if fh.Identity().Kind != source.Go {
		return nil, nil
	}
	ident, err := source.Identifier(ctx, snapshot, fh, params.Position, source.WidestPackageHandle)
	if err != nil {
		return nil, nil
	}
	h, err := ident.Hover(ctx)
	if err != nil {
		return nil, err
	}
	rng, err := ident.Range()
	if err != nil {
		return nil, err
	}
	hover, err := source.FormatHover(h, view.Options())
	if err != nil {
		return nil, err
	}
	return &protocol.Hover{
		Contents: protocol.MarkupContent{
			Kind:  view.Options().PreferredContentFormat,
			Value: hover,
		},
		Range: rng,
	}, nil
}
