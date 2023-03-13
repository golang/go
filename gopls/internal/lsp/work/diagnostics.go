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
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
)

func Diagnostics(ctx context.Context, snapshot source.Snapshot) (map[span.URI][]*source.Diagnostic, error) {
	ctx, done := event.Start(ctx, "work.Diagnostics", source.SnapshotLabels(snapshot)...)
	defer done()

	reports := map[span.URI][]*source.Diagnostic{}
	uri := snapshot.WorkFile()
	if uri == "" {
		return nil, nil
	}
	fh, err := snapshot.ReadFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	reports[fh.URI()] = []*source.Diagnostic{}
	diagnostics, err := DiagnosticsForWork(ctx, snapshot, fh)
	if err != nil {
		return nil, err
	}
	for _, d := range diagnostics {
		fh, err := snapshot.ReadFile(ctx, d.URI)
		if err != nil {
			return nil, err
		}
		reports[fh.URI()] = append(reports[fh.URI()], d)
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
		rng, err := pw.Mapper.OffsetRange(use.Syntax.Start.Byte, use.Syntax.End.Byte)
		if err != nil {
			return nil, err
		}

		modfh, err := snapshot.ReadFile(ctx, modFileURI(pw, use))
		if err != nil {
			return nil, err
		}
		if _, err := modfh.Content(); err != nil && os.IsNotExist(err) {
			diagnostics = append(diagnostics, &source.Diagnostic{
				URI:      fh.URI(),
				Range:    rng,
				Severity: protocol.SeverityError,
				Source:   source.WorkFileError,
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
