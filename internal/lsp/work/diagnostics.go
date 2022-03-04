// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func Diagnostics(ctx context.Context, snapshot source.Snapshot) (map[source.VersionedFileIdentity][]*source.Diagnostic, error) {
	ctx, done := event.Start(ctx, "work.Diagnostics", tag.Snapshot.Of(snapshot.ID()))
	defer done()

	reports := map[source.VersionedFileIdentity][]*source.Diagnostic{}
	uri := snapshot.WorkFile()
	if uri == "" {
		return nil, nil
	}
	fh, err := snapshot.GetVersionedFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	reports[fh.VersionedFileIdentity()] = []*source.Diagnostic{}
	diagnostics, err := DiagnosticsForWork(ctx, snapshot, fh)
	if err != nil {
		return nil, err
	}
	for _, d := range diagnostics {
		fh, err := snapshot.GetVersionedFile(ctx, d.URI)
		if err != nil {
			return nil, err
		}
		reports[fh.VersionedFileIdentity()] = append(reports[fh.VersionedFileIdentity()], d)
	}

	return reports, nil
}

func DiagnosticsForWork(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]*source.Diagnostic, error) {
	pw, err := snapshot.ParseWork(ctx, fh)
	if err != nil {
		if pw == nil || len(pw.ParseErrors) == 0 {
			return nil, err
		}
		return pw.ParseErrors, nil
	}

	// Add diagnostic if a directory does not contain a module.
	var diagnostics []*source.Diagnostic
	for _, use := range pw.File.Use {
		rng, err := source.LineToRange(pw.Mapper, fh.URI(), use.Syntax.Start, use.Syntax.End)
		if err != nil {
			return nil, err
		}

		modfh, err := snapshot.GetFile(ctx, modFileURI(pw, use))
		if err != nil {
			return nil, err
		}
		if _, err := modfh.Read(); err != nil && os.IsNotExist(err) {
			diagnostics = append(diagnostics, &source.Diagnostic{
				URI:      fh.URI(),
				Range:    rng,
				Severity: protocol.SeverityError,
				Source:   source.UnknownError, // Do we need a new source for this?
				Message:  fmt.Sprintf("directory %v does not contain a module", use.Path),
			})
		}
	}
	return diagnostics, nil
}

func modFileURI(pw *source.ParsedWorkFile, use *modfile.Use) span.URI {
	workdir := filepath.Dir(pw.URI.Filename())

	modroot := filepath.FromSlash(use.Path)
	if !filepath.IsAbs(modroot) {
		modroot = filepath.Join(workdir, modroot)
	}

	return span.URIFromPath(filepath.Join(modroot, "go.mod"))
}
