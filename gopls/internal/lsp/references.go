// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/template"
	"golang.org/x/tools/gopls/internal/telemetry"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
)

func (s *Server) references(ctx context.Context, params *protocol.ReferenceParams) (_ []protocol.Location, rerr error) {
	recordLatency := telemetry.StartLatencyTimer("references")
	defer func() {
		recordLatency(ctx, rerr)
	}()

	ctx, done := event.Start(ctx, "lsp.Server.references", tag.URI.Of(params.TextDocument.URI))
	defer done()

	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	defer release()
	if !ok {
		return nil, err
	}
	if snapshot.FileKind(fh) == source.Tmpl {
		return template.References(ctx, snapshot, fh, params)
	}
	return source.References(ctx, snapshot, fh, params.Position, params.Context.IncludeDeclaration)
}
