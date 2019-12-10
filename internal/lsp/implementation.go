// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
)

func (s *Server) implementation(ctx context.Context, params *protocol.ImplementationParams) ([]protocol.Location, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return nil, err
	}
	snapshot := view.Snapshot()
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	if f.Kind() != source.Go {
		return nil, nil
	}
	phs, err := snapshot.PackageHandles(ctx, snapshot.Handle(ctx, f))
	if err != nil {
		return nil, err
	}
	var (
		allLocs []protocol.Location
		seen    = make(map[protocol.Location]bool)
	)
	for _, ph := range phs {
		ctx := telemetry.Package.With(ctx, ph.ID())

		ident, err := source.Identifier(ctx, snapshot, f, params.Position, source.SpecificPackageHandle(ph.ID()))
		if err != nil {
			if err == source.ErrNoIdentFound {
				return nil, err
			}
			log.Error(ctx, "failed to find Identifer", err)
			continue
		}

		locs, err := ident.Implementation(ctx)
		if err != nil {
			if err == source.ErrNotAMethod {
				return nil, err
			}
			log.Error(ctx, "failed to find Implemenation", err)
			continue
		}

		for _, loc := range locs {
			if seen[loc] {
				continue
			}
			seen[loc] = true
			allLocs = append(allLocs, loc)
		}
	}

	return allLocs, nil
}
