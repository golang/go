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
	"golang.org/x/tools/internal/telemetry/trace"
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
			gof, _ := view.FindFile(ctx, uri).(source.GoFile)

			// If we have never seen this file before, there is nothing to do.
			if gof == nil {
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

				s.session.DidChangeOutOfBand(ctx, gof, change.Type)

				// Refresh diagnostics to reflect updated file contents.
				go func(view source.View) {
					ctx := view.BackgroundContext()
					ctx, done := trace.StartSpan(ctx, "lsp:background-worker")
					defer done()
					s.Diagnostics(ctx, view, uri)
				}(view)
			case protocol.Created:
				log.Print(ctx, "watched file created", telemetry.File)
			case protocol.Deleted:
				log.Print(ctx, "watched file deleted", telemetry.File)

				pkg, err := gof.GetPackage(ctx)
				if err != nil {
					log.Error(ctx, "didChangeWatchedFiles: GetPackage", err, telemetry.File)
					continue
				}

				// Find a different file in the same package we can use to
				// trigger diagnostics.
				var otherFile source.GoFile
				for _, pgh := range pkg.GetHandles() {
					ident := pgh.File().Identity()
					if ident.URI == gof.URI() {
						continue
					}
					otherFile, _ = view.FindFile(ctx, ident.URI).(source.GoFile)
					if otherFile != nil {
						break
					}
				}

				s.session.DidChangeOutOfBand(ctx, gof, change.Type)

				if otherFile != nil {
					// Refresh diagnostics to reflect updated file contents.
					go func(view source.View) {
						ctx := view.BackgroundContext()
						ctx, done := trace.StartSpan(ctx, "lsp:background-worker")
						defer done()
						s.Diagnostics(ctx, view, otherFile.URI())
					}(view)
				} else {
					// TODO: Handle case when there is no other file (i.e. deleted
					//       file was the only file in the package).
				}
			}
		}
	}

	return nil
}
