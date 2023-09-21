// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/telemetry"
	"golang.org/x/tools/internal/event"
)

func (s *Server) symbol(ctx context.Context, params *protocol.WorkspaceSymbolParams) (_ []protocol.SymbolInformation, rerr error) {
	recordLatency := telemetry.StartLatencyTimer("symbol")
	defer func() {
		recordLatency(ctx, rerr)
	}()

	ctx, done := event.Start(ctx, "lsp.Server.symbol")
	defer done()

	views := s.session.Views()
	matcher := s.Options().SymbolMatcher
	style := s.Options().SymbolStyle
	// TODO(rfindley): it looks wrong that we need to pass views here.
	//
	// Evidence:
	//  - this is the only place we convert views to []source.View
	//  - workspace symbols is the only place where we call source.View.Snapshot
	var sourceViews []source.View
	for _, v := range views {
		sourceViews = append(sourceViews, v)
	}
	return source.WorkspaceSymbols(ctx, matcher, style, sourceViews, params.Query)
}
