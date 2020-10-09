// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mod provides core features related to go.mod file
// handling for use by Go editors and tools.
package mod

import (
	"context"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func Diagnostics(ctx context.Context, snapshot source.Snapshot) (map[source.VersionedFileIdentity][]*source.Diagnostic, error) {
	ctx, done := event.Start(ctx, "mod.Diagnostics", tag.Snapshot.Of(snapshot.ID()))
	defer done()

	reports := map[source.VersionedFileIdentity][]*source.Diagnostic{}
	for _, uri := range snapshot.ModFiles() {
		fh, err := snapshot.GetFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		reports[fh.VersionedFileIdentity()] = []*source.Diagnostic{}
		tidied, err := snapshot.ModTidy(ctx, fh)
		if err == source.ErrTmpModfileUnsupported {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		for _, e := range tidied.Errors {
			diag := &source.Diagnostic{
				Message: e.Message,
				Range:   e.Range,
				Source:  e.Category,
			}
			if e.Category == "syntax" {
				diag.Severity = protocol.SeverityError
			} else {
				diag.Severity = protocol.SeverityWarning
			}
			fh, err := snapshot.GetFile(ctx, e.URI)
			if err != nil {
				return nil, err
			}
			reports[fh.VersionedFileIdentity()] = append(reports[fh.VersionedFileIdentity()], diag)
		}
	}
	return reports, nil
}
