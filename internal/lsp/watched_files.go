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

func (s *Server) didChangeWatchedFiles(ctx context.Context, params *protocol.DidChangeWatchedFilesParams) error {
	for _, change := range params.Changes {
		uri := span.NewURI(change.URI)
		ctx := telemetry.File.With(ctx, uri)

		for _, view := range s.session.Views() {
			if !view.Options().WatchFileChanges {
				continue
			}
			switch change.Type {
			case protocol.Changed, protocol.Created:
				// If client has this file open, don't do anything.
				// The client's contents must remain the source of truth.
				if s.session.IsOpen(uri) {
					break
				}
				if s.session.DidChangeOutOfBand(ctx, uri, change.Type) {
					// If we had been tracking the given file,
					// recompute diagnostics to reflect updated file contents.
					go s.diagnostics(view, uri)
				}
			case protocol.Deleted:
				f := view.FindFile(ctx, uri)
				// If we have never seen this file before, there is nothing to do.
				if f == nil {
					continue
				}
				_, cphs, err := view.CheckPackageHandles(ctx, f)
				if err != nil {
					log.Error(ctx, "didChangeWatchedFiles: CheckPackageHandles", err, telemetry.File)
					continue
				}
				cph, err := source.WidestCheckPackageHandle(cphs)
				if err != nil {
					log.Error(ctx, "didChangeWatchedFiles: WidestCheckPackageHandle", err, telemetry.File)
					continue
				}
				// Find a different file in the same package we can use to trigger diagnostics.
				// TODO(rstambler): Allow diagnostics to be called per-package to avoid this.
				var otherFile source.File
				for _, ph := range cph.Files() {
					if ph.File().Identity().URI == f.URI() {
						continue
					}
					if f := view.FindFile(ctx, ph.File().Identity().URI); f != nil && s.session.IsOpen(f.URI()) {
						otherFile = f
						break
					}
				}

				// Notify the view of the deletion of the file.
				s.session.DidChangeOutOfBand(ctx, uri, change.Type)

				// If this was the only file in the package, clear its diagnostics.
				if otherFile == nil {
					if err := s.publishDiagnostics(ctx, uri, []source.Diagnostic{}); err != nil {
						log.Error(ctx, "failed to clear diagnostics", err, telemetry.URI.Of(uri))
					}
					return nil
				}

				// Refresh diagnostics for the package the file belonged to.
				go s.diagnostics(view, otherFile.URI())
			}
		}
	}
	return nil
}
