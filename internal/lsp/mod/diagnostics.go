// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mod provides core features related to go.mod file
// handling for use by Go editors and tools.
package mod

import (
	"context"
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func Diagnostics(ctx context.Context, snapshot source.Snapshot) (map[source.FileIdentity][]*source.Diagnostic, error) {
	uri := snapshot.View().ModFile()
	if uri == "" {
		return nil, nil
	}

	ctx, done := event.Start(ctx, "mod.Diagnostics", tag.URI.Of(uri))
	defer done()

	fh, err := snapshot.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	mth, err := snapshot.ModTidyHandle(ctx)
	if err == source.ErrTmpModfileUnsupported {
		return nil, nil
	}
	reports := map[source.FileIdentity][]*source.Diagnostic{
		fh.Identity(): {},
	}
	if err != nil {
		return nil, err
	}
	diagnostics, err := mth.Tidy(ctx)
	if err != nil {
		return nil, err
	}
	for _, e := range diagnostics {
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
		reports[fh.Identity()] = append(reports[fh.Identity()], diag)
	}
	return reports, nil
}

var moduleAtVersionRe = regexp.MustCompile(`^(?P<module>.*)@(?P<version>.*)$`)

// ExtractGoCommandError tries to parse errors that come from the go command
// and shape them into go.mod diagnostics.
func ExtractGoCommandError(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle, loadErr error) (*source.Diagnostic, error) {
	// We try to match module versions in error messages. Some examples:
	//
	//  example.com@v1.2.2: reading example.com/@v/v1.2.2.mod: no such file or directory
	//  go: github.com/cockroachdb/apd/v2@v2.0.72: reading github.com/cockroachdb/apd/go.mod at revision v2.0.72: unknown revision v2.0.72
	//  go: example.com@v1.2.3 requires\n\trandom.org@v1.2.3: parsing go.mod:\n\tmodule declares its path as: bob.org\n\tbut was required as: random.org
	//
	// We split on colons and whitespace, and attempt to match on something
	// that matches module@version. If we're able to find a match, we try to
	// find anything that matches it in the go.mod file.
	var v module.Version
	fields := strings.FieldsFunc(loadErr.Error(), func(r rune) bool {
		return unicode.IsSpace(r) || r == ':'
	})
	for _, s := range fields {
		s = strings.TrimSpace(s)
		match := moduleAtVersionRe.FindStringSubmatch(s)
		if match == nil || len(match) < 3 {
			continue
		}
		v.Path = match[1]
		v.Version = match[2]
		if err := module.Check(v.Path, v.Version); err == nil {
			break
		}
	}
	pmh, err := snapshot.ParseModHandle(ctx, fh)
	if err != nil {
		return nil, err
	}
	parsed, m, _, err := pmh.Parse(ctx)
	if err != nil {
		return nil, err
	}
	toDiagnostic := func(line *modfile.Line) (*source.Diagnostic, error) {
		rng, err := rangeFromPositions(fh.URI(), m, line.Start, line.End)
		if err != nil {
			return nil, err
		}
		return &source.Diagnostic{
			Message:  loadErr.Error(),
			Range:    rng,
			Severity: protocol.SeverityError,
		}, nil
	}
	// Check if there are any require, exclude, or replace statements that
	// match this module version.
	for _, req := range parsed.Require {
		if req.Mod != v {
			continue
		}
		return toDiagnostic(req.Syntax)
	}
	for _, ex := range parsed.Exclude {
		if ex.Mod != v {
			continue
		}
		return toDiagnostic(ex.Syntax)
	}
	for _, rep := range parsed.Replace {
		if rep.New != v && rep.Old != v {
			continue
		}
		return toDiagnostic(rep.Syntax)
	}
	// No match for the module path was found in the go.mod file.
	// Show the error on the module declaration.
	return toDiagnostic(parsed.Module.Syntax)
}

func rangeFromPositions(uri span.URI, m *protocol.ColumnMapper, s, e modfile.Position) (protocol.Range, error) {
	toPoint := func(offset int) (span.Point, error) {
		l, c, err := m.Converter.ToPosition(offset)
		if err != nil {
			return span.Point{}, err
		}
		return span.NewPoint(l, c, offset), nil
	}
	start, err := toPoint(s.Byte)
	if err != nil {
		return protocol.Range{}, err
	}
	end, err := toPoint(e.Byte)
	if err != nil {
		return protocol.Range{}, err
	}
	return m.Range(span.New(uri, start, end))
}
