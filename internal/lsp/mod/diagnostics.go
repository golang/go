// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mod provides core features related to go.mod file
// handling for use by Go editors and tools.
package mod

import (
	"context"
	"fmt"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/telemetry/trace"
)

func Diagnostics(ctx context.Context, snapshot source.Snapshot) (map[source.FileIdentity][]source.Diagnostic, error) {
	// TODO: We will want to support diagnostics for go.mod files even when the -modfile flag is turned off.
	realfh, tempfh, err := snapshot.ModFiles(ctx)
	if err != nil {
		return nil, err
	}
	// Check the case when the tempModfile flag is turned off.
	if realfh == nil || tempfh == nil {
		return nil, nil
	}
	ctx, done := trace.StartSpan(ctx, "modfiles.Diagnostics", telemetry.File.Of(realfh.Identity().URI))
	defer done()

	_, _, parseErrors, err := snapshot.ParseModHandle(ctx, realfh).Parse(ctx)
	if err != nil {
		return nil, err
	}

	reports := map[source.FileIdentity][]source.Diagnostic{
		realfh.Identity(): []source.Diagnostic{},
	}
	for _, e := range parseErrors {
		diag := source.Diagnostic{
			Message:        e.Message,
			Range:          e.Range,
			SuggestedFixes: e.SuggestedFixes,
			Source:         e.Category,
		}
		if e.Category == "syntax" {
			diag.Severity = protocol.SeverityError
		} else {
			diag.Severity = protocol.SeverityWarning
		}
		reports[realfh.Identity()] = append(reports[realfh.Identity()], diag)
	}
	return reports, nil
}

// TODO: Add caching for go.mod diagnostics to be able to map them back to source.Diagnostics
// and reuse the cached suggested fixes.
func SuggestedFixes(fh source.FileHandle, diags []protocol.Diagnostic) []protocol.CodeAction {
	var actions []protocol.CodeAction
	for _, diag := range diags {
		var title string
		if strings.Contains(diag.Message, "is not used in this module") {
			split := strings.Split(diag.Message, " ")
			if len(split) < 1 {
				continue
			}
			title = fmt.Sprintf("Remove dependency: %s", split[0])
		}
		if strings.Contains(diag.Message, "should be a direct dependency.") {
			title = "Remove indirect"
		}
		if title == "" {
			continue
		}
		actions = append(actions, protocol.CodeAction{
			Title: title,
			Kind:  protocol.QuickFix,
			Edit: protocol.WorkspaceEdit{
				DocumentChanges: []protocol.TextDocumentEdit{
					{
						TextDocument: protocol.VersionedTextDocumentIdentifier{
							Version: fh.Identity().Version,
							TextDocumentIdentifier: protocol.TextDocumentIdentifier{
								URI: protocol.NewURI(fh.Identity().URI),
							},
						},
						Edits: []protocol.TextEdit{protocol.TextEdit{Range: diag.Range, NewText: ""}},
					},
				},
			},
			Diagnostics: diags,
		})
	}
	return actions
}
