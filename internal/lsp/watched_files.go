// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"sort"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
)

func (s *Server) didChangeWatchedFiles(ctx context.Context, params *protocol.DidChangeWatchedFilesParams) error {
	options := s.session.Options()
	if !options.WatchFileChanges {
		return nil
	}

	for _, change := range params.Changes {
		uri := span.NewURI(change.URI)

		ctx := telemetry.File.With(ctx, uri)

		for _, view := range s.session.Views() {
			f := view.FindFile(ctx, uri)

			// If we have never seen this file before, there is nothing to do.
			if f == nil {
				continue
			}

			// If client has this file open, don't do anything. The client's contents
			// must remain the source of truth.
			if s.session.IsOpen(uri) {
				break
			}

			switch change.Type {
			case protocol.Changed:
				log.Print(ctx, "watched file changed", telemetry.File)

				s.session.DidChangeOutOfBand(ctx, uri, change.Type)

				// Refresh diagnostics to reflect updated file contents.
				go s.diagnostics(view, uri)
			case protocol.Created:
				log.Print(ctx, "watched file created", telemetry.File)
			case protocol.Deleted:
				log.Print(ctx, "watched file deleted", telemetry.File)

				_, cphs, err := view.CheckPackageHandles(ctx, f)
				if err != nil {
					log.Error(ctx, "didChangeWatchedFiles: GetPackage", err, telemetry.File)
					continue
				}
				// Find a different file in the same package we can use to trigger diagnostics.
				// TODO(rstambler): Allow diagnostics to be called per-package to avoid this.
				var otherFile source.File
				sort.Slice(cphs, func(i, j int) bool {
					return len(cphs[i].Files()) > len(cphs[j].Files())
				})
				for _, ph := range cphs[0].Files() {
					if len(cphs) > 1 && contains(cphs[1], ph.File()) {
						continue
					}
					ident := ph.File().Identity()
					if ident.URI == f.URI() {
						continue
					}
					otherFile := view.FindFile(ctx, ident.URI)
					if otherFile != nil {
						break
					}
				}
				s.session.DidChangeOutOfBand(ctx, uri, change.Type)

				// If this was the only file in the package, clear its diagnostics.
				if otherFile == nil {
					if err := s.publishDiagnostics(ctx, uri, []source.Diagnostic{}); err != nil {
						log.Error(ctx, "failed to clear diagnostics", err, telemetry.URI.Of(uri))
					}
					return nil
				}
				go s.diagnostics(view, otherFile.URI())
			}
		}
	}
	return nil
}

func contains(cph source.CheckPackageHandle, fh source.FileHandle) bool {
	for _, ph := range cph.Files() {
		if ph.File().Identity().URI == fh.Identity().URI {
			return true
		}
	}
	return false
}
