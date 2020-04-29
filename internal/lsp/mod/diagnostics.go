// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mod provides core features related to go.mod file
// handling for use by Go editors and tools.
package mod

import (
	"context"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func Diagnostics(ctx context.Context, snapshot source.Snapshot) (map[source.FileIdentity][]*source.Diagnostic, map[string]*modfile.Require, error) {
	// TODO: We will want to support diagnostics for go.mod files even when the -modfile flag is turned off.
	realURI, tempURI := snapshot.View().ModFiles()

	// Check the case when the tempModfile flag is turned off.
	if realURI == "" || tempURI == "" {
		return nil, nil, nil
	}
	ctx, done := event.Start(ctx, "mod.Diagnostics", tag.URI.Of(realURI))
	defer done()

	realfh, err := snapshot.GetFile(realURI)
	if err != nil {
		return nil, nil, err
	}
	mth, err := snapshot.ModTidyHandle(ctx, realfh)
	if err != nil {
		return nil, nil, err
	}
	_, _, missingDeps, parseErrors, err := mth.Tidy(ctx)
	if err != nil {
		return nil, nil, err
	}
	reports := map[source.FileIdentity][]*source.Diagnostic{
		realfh.Identity(): {},
	}
	for _, e := range parseErrors {
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
		reports[realfh.Identity()] = append(reports[realfh.Identity()], diag)
	}
	return reports, missingDeps, nil
}

func SuggestedFixes(ctx context.Context, snapshot source.Snapshot, realfh source.FileHandle, diags []protocol.Diagnostic) ([]protocol.CodeAction, error) {
	mth, err := snapshot.ModTidyHandle(ctx, realfh)
	if err != nil {
		return nil, err
	}
	_, _, _, parseErrors, err := mth.Tidy(ctx)
	if err != nil {
		return nil, err
	}
	errorsMap := make(map[string][]source.Error)
	for _, e := range parseErrors {
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
					fh, err := snapshot.GetFile(uri)
					if err != nil {
						return nil, err
					}
					action.Edit.DocumentChanges = append(action.Edit.DocumentChanges, protocol.TextDocumentEdit{
						TextDocument: protocol.VersionedTextDocumentIdentifier{
							Version: fh.Identity().Version,
							TextDocumentIdentifier: protocol.TextDocumentIdentifier{
								URI: protocol.URIFromSpanURI(fh.Identity().URI),
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

func SuggestedGoFixes(ctx context.Context, snapshot source.Snapshot) (map[string]protocol.TextDocumentEdit, error) {
	// TODO(rstambler): Support diagnostics for go.mod files even when the
	// -modfile flag is turned off.
	realURI, tempURI := snapshot.View().ModFiles()
	if realURI == "" || tempURI == "" {
		return nil, nil
	}

	ctx, done := event.Start(ctx, "mod.SuggestedGoFixes", tag.URI.Of(realURI))
	defer done()

	realfh, err := snapshot.GetFile(realURI)
	if err != nil {
		return nil, err
	}
	mth, err := snapshot.ModTidyHandle(ctx, realfh)
	if err != nil {
		return nil, err
	}
	realFile, realMapper, missingDeps, _, err := mth.Tidy(ctx)
	if err != nil {
		return nil, err
	}
	if len(missingDeps) == 0 {
		return nil, nil
	}
	// Get the contents of the go.mod file before we make any changes.
	oldContents, _, err := realfh.Read(ctx)
	if err != nil {
		return nil, err
	}
	textDocumentEdits := make(map[string]protocol.TextDocumentEdit)
	for dep, req := range missingDeps {
		// Calculate the quick fix edits that need to be made to the go.mod file.
		if err := realFile.AddRequire(req.Mod.Path, req.Mod.Version); err != nil {
			return nil, err
		}
		realFile.Cleanup()
		newContents, err := realFile.Format()
		if err != nil {
			return nil, err
		}
		// Reset the *modfile.File back to before we added the dependency.
		if err := realFile.DropRequire(req.Mod.Path); err != nil {
			return nil, err
		}
		// Calculate the edits to be made due to the change.
		diff := snapshot.View().Options().ComputeEdits(realfh.Identity().URI, string(oldContents), string(newContents))
		edits, err := source.ToProtocolEdits(realMapper, diff)
		if err != nil {
			return nil, err
		}
		textDocumentEdits[dep] = protocol.TextDocumentEdit{
			TextDocument: protocol.VersionedTextDocumentIdentifier{
				Version: realfh.Identity().Version,
				TextDocumentIdentifier: protocol.TextDocumentIdentifier{
					URI: protocol.URIFromSpanURI(realfh.Identity().URI),
				},
			},
			Edits: edits,
		}
	}
	return textDocumentEdits, nil
}

func sameDiagnostic(d protocol.Diagnostic, e source.Error) bool {
	return d.Message == e.Message && protocol.CompareRange(d.Range, e.Range) == 0 && d.Source == e.Category
}
