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
		fh, err := snapshot.GetVersionedFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		reports[fh.VersionedFileIdentity()] = []*source.Diagnostic{}
		errors, err := ErrorsForMod(ctx, snapshot, fh)
		if err != nil {
			return nil, err
		}
		for _, e := range errors {
			d := &source.Diagnostic{
				Message: e.Message,
				Range:   e.Range,
				Source:  e.Category,
			}
			if e.Category == "syntax" {
				d.Severity = protocol.SeverityError
			} else {
				d.Severity = protocol.SeverityWarning
			}
			fh, err := snapshot.GetVersionedFile(ctx, e.URI)
			if err != nil {
				return nil, err
			}
			reports[fh.VersionedFileIdentity()] = append(reports[fh.VersionedFileIdentity()], d)
		}
	}
	return reports, nil
}

func ErrorsForMod(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]source.Error, error) {
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil {
		if pm == nil || len(pm.ParseErrors) == 0 {
			return nil, err
		}
		return pm.ParseErrors, nil
	}
	tidied, err := snapshot.ModTidy(ctx, pm)

	if source.IsNonFatalGoModError(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return tidied.Errors, nil
}
