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
	if err != nil {
		return nil, err
	}
	diagnostics, err := mth.Tidy(ctx)
	if err != nil {
		return nil, err
	}
	reports := map[source.FileIdentity][]*source.Diagnostic{
		fh.Identity(): {},
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

func SuggestedFixes(ctx context.Context, snapshot source.Snapshot, diags []protocol.Diagnostic) ([]protocol.CodeAction, error) {
	mth, err := snapshot.ModTidyHandle(ctx)
	if err == source.ErrTmpModfileUnsupported {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	diagnostics, err := mth.Tidy(ctx)
	if err != nil {
		return nil, err
	}
	errorsMap := make(map[string][]source.Error)
	for _, e := range diagnostics {
		if errorsMap[e.Message] == nil {
			errorsMap[e.Message] = []source.Error{}
		}
		errorsMap[e.Message] = append(errorsMap[e.Message], e)
	}
	var actions []protocol.CodeAction
	for _, diag := range diags {
		for _, e := range errorsMap[diag.Message] {
			if !sameDiagnostic(diag, e) {
				continue
			}
			for _, fix := range e.SuggestedFixes {
				action := protocol.CodeAction{
					Title:       fix.Title,
					Kind:        protocol.QuickFix,
					Diagnostics: []protocol.Diagnostic{diag},
					Edit:        protocol.WorkspaceEdit{},
				}
				for uri, edits := range fix.Edits {
					fh, err := snapshot.GetFile(ctx, uri)
					if err != nil {
						return nil, err
					}
					action.Edit.DocumentChanges = append(action.Edit.DocumentChanges, protocol.TextDocumentEdit{
						TextDocument: protocol.VersionedTextDocumentIdentifier{
							Version: fh.Version(),
							TextDocumentIdentifier: protocol.TextDocumentIdentifier{
								URI: protocol.URIFromSpanURI(fh.URI()),
							},
						},
						Edits: edits,
					})
				}
				actions = append(actions, action)
			}
		}
	}
	return actions, nil
}

func sameDiagnostic(d protocol.Diagnostic, e source.Error) bool {
	return d.Message == e.Message && protocol.CompareRange(d.Range, e.Range) == 0 && d.Source == e.Category
}
