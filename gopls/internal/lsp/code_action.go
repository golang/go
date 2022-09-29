// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/mod"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/internal/imports"
)

func (s *Server) codeAction(ctx context.Context, params *protocol.CodeActionParams) ([]protocol.CodeAction, error) {
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	defer release()
	if !ok {
		return nil, err
	}
	uri := fh.URI()

	// Determine the supported actions for this file kind.
	kind := snapshot.View().FileKind(fh)
	supportedCodeActions, ok := snapshot.View().Options().SupportedCodeActions[kind]
	if !ok {
		return nil, fmt.Errorf("no supported code actions for %v file kind", kind)
	}

	// The Only field of the context specifies which code actions the client wants.
	// If Only is empty, assume that the client wants all of the non-explicit code actions.
	var wanted map[protocol.CodeActionKind]bool

	// Explicit Code Actions are opt-in and shouldn't be returned to the client unless
	// requested using Only.
	// TODO: Add other CodeLenses such as GoGenerate, RegenerateCgo, etc..
	explicit := map[protocol.CodeActionKind]bool{
		protocol.GoTest: true,
	}

	if len(params.Context.Only) == 0 {
		wanted = supportedCodeActions
	} else {
		wanted = make(map[protocol.CodeActionKind]bool)
		for _, only := range params.Context.Only {
			for k, v := range supportedCodeActions {
				if only == k || strings.HasPrefix(string(k), string(only)+".") {
					wanted[k] = wanted[k] || v
				}
			}
			wanted[only] = wanted[only] || explicit[only]
		}
	}
	if len(supportedCodeActions) == 0 {
		return nil, nil // not an error if there are none supported
	}
	if len(wanted) == 0 {
		return nil, fmt.Errorf("no supported code action to execute for %s, wanted %v", uri, params.Context.Only)
	}

	var codeActions []protocol.CodeAction
	switch kind {
	case source.Mod:
		if diagnostics := params.Context.Diagnostics; len(diagnostics) > 0 {
			diags, err := mod.ModDiagnostics(ctx, snapshot, fh)
			if source.IsNonFatalGoModError(err) {
				return nil, nil
			}
			if err != nil {
				return nil, err
			}
			udiags, err := mod.ModUpgradeDiagnostics(ctx, snapshot, fh)
			if err != nil {
				return nil, err
			}
			quickFixes, err := codeActionsMatchingDiagnostics(ctx, snapshot, diagnostics, append(diags, udiags...))
			if err != nil {
				return nil, err
			}
			codeActions = append(codeActions, quickFixes...)

			vdiags, err := mod.ModVulnerabilityDiagnostics(ctx, snapshot, fh)
			if err != nil {
				return nil, err
			}
			// Group vulnerabilities by location and then limit which code actions we return
			// for each location.
			m := make(map[protocol.Range][]*source.Diagnostic)
			for _, v := range vdiags {
				m[v.Range] = append(m[v.Range], v)
			}
			for _, sdiags := range m {
				quickFixes, err = codeActionsMatchingDiagnostics(ctx, snapshot, diagnostics, sdiags)
				if err != nil {
					return nil, err
				}
				quickFixes = mod.SelectUpgradeCodeActions(quickFixes)
				codeActions = append(codeActions, quickFixes...)
			}
		}
	case source.Go:
		// Don't suggest fixes for generated files, since they are generally
		// not useful and some editors may apply them automatically on save.
		if source.IsGenerated(ctx, snapshot, uri) {
			return nil, nil
		}
		diagnostics := params.Context.Diagnostics

		// First, process any missing imports and pair them with the
		// diagnostics they fix.
		if wantQuickFixes := wanted[protocol.QuickFix] && len(diagnostics) > 0; wantQuickFixes || wanted[protocol.SourceOrganizeImports] {
			importEdits, importEditsPerFix, err := source.AllImportsFixes(ctx, snapshot, fh)
			if err != nil {
				event.Error(ctx, "imports fixes", err, tag.File.Of(fh.URI().Filename()))
			}
			// Separate this into a set of codeActions per diagnostic, where
			// each action is the addition, removal, or renaming of one import.
			if wantQuickFixes {
				for _, importFix := range importEditsPerFix {
					fixes := importDiagnostics(importFix.Fix, diagnostics)
					if len(fixes) == 0 {
						continue
					}
					codeActions = append(codeActions, protocol.CodeAction{
						Title: importFixTitle(importFix.Fix),
						Kind:  protocol.QuickFix,
						Edit: protocol.WorkspaceEdit{
							DocumentChanges: documentChanges(fh, importFix.Edits),
						},
						Diagnostics: fixes,
					})
				}
			}

			// Send all of the import edits as one code action if the file is
			// being organized.
			if wanted[protocol.SourceOrganizeImports] && len(importEdits) > 0 {
				codeActions = append(codeActions, protocol.CodeAction{
					Title: "Organize Imports",
					Kind:  protocol.SourceOrganizeImports,
					Edit: protocol.WorkspaceEdit{
						DocumentChanges: documentChanges(fh, importEdits),
					},
				})
			}
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		pkg, err := snapshot.PackageForFile(ctx, fh.URI(), source.TypecheckFull, source.WidestPackage)
		if err != nil {
			return nil, err
		}

		pkgDiagnostics, err := snapshot.DiagnosePackage(ctx, pkg)
		if err != nil {
			return nil, err
		}
		analysisDiags, err := source.Analyze(ctx, snapshot, pkg, true)
		if err != nil {
			return nil, err
		}
		fileDiags := append(pkgDiagnostics[uri], analysisDiags[uri]...)

		// Split diagnostics into fixes, which must match incoming diagnostics,
		// and non-fixes, which must match the requested range. Build actions
		// for all of them.
		var fixDiags, nonFixDiags []*source.Diagnostic
		for _, d := range fileDiags {
			if len(d.SuggestedFixes) == 0 {
				continue
			}
			var isFix bool
			for _, fix := range d.SuggestedFixes {
				if fix.ActionKind == protocol.QuickFix || fix.ActionKind == protocol.SourceFixAll {
					isFix = true
					break
				}
			}
			if isFix {
				fixDiags = append(fixDiags, d)
			} else {
				nonFixDiags = append(nonFixDiags, d)
			}
		}

		fixActions, err := codeActionsMatchingDiagnostics(ctx, snapshot, diagnostics, fixDiags)
		if err != nil {
			return nil, err
		}
		codeActions = append(codeActions, fixActions...)

		for _, nonfix := range nonFixDiags {
			// For now, only show diagnostics for matching lines. Maybe we should
			// alter this behavior in the future, depending on the user experience.
			if !protocol.Intersect(nonfix.Range, params.Range) {
				continue
			}
			actions, err := codeActionsForDiagnostic(ctx, snapshot, nonfix, nil)
			if err != nil {
				return nil, err
			}
			codeActions = append(codeActions, actions...)
		}

		if wanted[protocol.RefactorExtract] {
			fixes, err := extractionFixes(ctx, snapshot, pkg, uri, params.Range)
			if err != nil {
				return nil, err
			}
			codeActions = append(codeActions, fixes...)
		}

		if wanted[protocol.GoTest] {
			fixes, err := goTest(ctx, snapshot, uri, params.Range)
			if err != nil {
				return nil, err
			}
			codeActions = append(codeActions, fixes...)
		}

	default:
		// Unsupported file kind for a code action.
		return nil, nil
	}

	var filtered []protocol.CodeAction
	for _, action := range codeActions {
		if wanted[action.Kind] {
			filtered = append(filtered, action)
		}
	}
	return filtered, nil
}

func (s *Server) getSupportedCodeActions() []protocol.CodeActionKind {
	allCodeActionKinds := make(map[protocol.CodeActionKind]struct{})
	for _, kinds := range s.session.Options().SupportedCodeActions {
		for kind := range kinds {
			allCodeActionKinds[kind] = struct{}{}
		}
	}
	var result []protocol.CodeActionKind
	for kind := range allCodeActionKinds {
		result = append(result, kind)
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i] < result[j]
	})
	return result
}

func importFixTitle(fix *imports.ImportFix) string {
	var str string
	switch fix.FixType {
	case imports.AddImport:
		str = fmt.Sprintf("Add import: %s %q", fix.StmtInfo.Name, fix.StmtInfo.ImportPath)
	case imports.DeleteImport:
		str = fmt.Sprintf("Delete import: %s %q", fix.StmtInfo.Name, fix.StmtInfo.ImportPath)
	case imports.SetImportName:
		str = fmt.Sprintf("Rename import: %s %q", fix.StmtInfo.Name, fix.StmtInfo.ImportPath)
	}
	return str
}

func importDiagnostics(fix *imports.ImportFix, diagnostics []protocol.Diagnostic) (results []protocol.Diagnostic) {
	for _, diagnostic := range diagnostics {
		switch {
		// "undeclared name: X" may be an unresolved import.
		case strings.HasPrefix(diagnostic.Message, "undeclared name: "):
			ident := strings.TrimPrefix(diagnostic.Message, "undeclared name: ")
			if ident == fix.IdentName {
				results = append(results, diagnostic)
			}
		// "undefined: X" may be an unresolved import at Go 1.20+.
		case strings.HasPrefix(diagnostic.Message, "undefined: "):
			ident := strings.TrimPrefix(diagnostic.Message, "undefined: ")
			if ident == fix.IdentName {
				results = append(results, diagnostic)
			}
		// "could not import: X" may be an invalid import.
		case strings.HasPrefix(diagnostic.Message, "could not import: "):
			ident := strings.TrimPrefix(diagnostic.Message, "could not import: ")
			if ident == fix.IdentName {
				results = append(results, diagnostic)
			}
		// "X imported but not used" is an unused import.
		// "X imported but not used as Y" is an unused import.
		case strings.Contains(diagnostic.Message, " imported but not used"):
			idx := strings.Index(diagnostic.Message, " imported but not used")
			importPath := diagnostic.Message[:idx]
			if importPath == fmt.Sprintf("%q", fix.StmtInfo.ImportPath) {
				results = append(results, diagnostic)
			}
		}
	}
	return results
}

func extractionFixes(ctx context.Context, snapshot source.Snapshot, pkg source.Package, uri span.URI, rng protocol.Range) ([]protocol.CodeAction, error) {
	if rng.Start == rng.End {
		return nil, nil
	}
	fh, err := snapshot.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	_, pgf, err := source.GetParsedFile(ctx, snapshot, fh, source.NarrowestPackage)
	if err != nil {
		return nil, fmt.Errorf("getting file for Identifier: %w", err)
	}
	srng, err := pgf.Mapper.RangeToSpanRange(rng)
	if err != nil {
		return nil, err
	}
	puri := protocol.URIFromSpanURI(uri)
	var commands []protocol.Command
	if _, ok, methodOk, _ := source.CanExtractFunction(pgf.Tok, srng, pgf.Src, pgf.File); ok {
		cmd, err := command.NewApplyFixCommand("Extract function", command.ApplyFixArgs{
			URI:   puri,
			Fix:   source.ExtractFunction,
			Range: rng,
		})
		if err != nil {
			return nil, err
		}
		commands = append(commands, cmd)
		if methodOk {
			cmd, err := command.NewApplyFixCommand("Extract method", command.ApplyFixArgs{
				URI:   puri,
				Fix:   source.ExtractMethod,
				Range: rng,
			})
			if err != nil {
				return nil, err
			}
			commands = append(commands, cmd)
		}
	}
	if _, _, ok, _ := source.CanExtractVariable(srng, pgf.File); ok {
		cmd, err := command.NewApplyFixCommand("Extract variable", command.ApplyFixArgs{
			URI:   puri,
			Fix:   source.ExtractVariable,
			Range: rng,
		})
		if err != nil {
			return nil, err
		}
		commands = append(commands, cmd)
	}
	var actions []protocol.CodeAction
	for i := range commands {
		actions = append(actions, protocol.CodeAction{
			Title:   commands[i].Title,
			Kind:    protocol.RefactorExtract,
			Command: &commands[i],
		})
	}
	return actions, nil
}

func documentChanges(fh source.VersionedFileHandle, edits []protocol.TextEdit) []protocol.DocumentChanges {
	return []protocol.DocumentChanges{
		{
			TextDocumentEdit: &protocol.TextDocumentEdit{
				TextDocument: protocol.OptionalVersionedTextDocumentIdentifier{
					Version: fh.Version(),
					TextDocumentIdentifier: protocol.TextDocumentIdentifier{
						URI: protocol.URIFromSpanURI(fh.URI()),
					},
				},
				Edits: edits,
			},
		},
	}
}

func codeActionsMatchingDiagnostics(ctx context.Context, snapshot source.Snapshot, pdiags []protocol.Diagnostic, sdiags []*source.Diagnostic) ([]protocol.CodeAction, error) {
	var actions []protocol.CodeAction
	for _, sd := range sdiags {
		var diag *protocol.Diagnostic
		for _, pd := range pdiags {
			if sameDiagnostic(pd, sd) {
				diag = &pd
				break
			}
		}
		if diag == nil {
			continue
		}
		diagActions, err := codeActionsForDiagnostic(ctx, snapshot, sd, diag)
		if err != nil {
			return nil, err
		}
		actions = append(actions, diagActions...)

	}
	return actions, nil
}

func codeActionsForDiagnostic(ctx context.Context, snapshot source.Snapshot, sd *source.Diagnostic, pd *protocol.Diagnostic) ([]protocol.CodeAction, error) {
	var actions []protocol.CodeAction
	for _, fix := range sd.SuggestedFixes {
		var changes []protocol.DocumentChanges
		for uri, edits := range fix.Edits {
			fh, err := snapshot.GetVersionedFile(ctx, uri)
			if err != nil {
				return nil, err
			}
			changes = append(changes, protocol.DocumentChanges{
				TextDocumentEdit: &protocol.TextDocumentEdit{
					TextDocument: protocol.OptionalVersionedTextDocumentIdentifier{
						Version: fh.Version(),
						TextDocumentIdentifier: protocol.TextDocumentIdentifier{
							URI: protocol.URIFromSpanURI(fh.URI()),
						},
					},
					Edits: edits,
				},
			})
		}
		action := protocol.CodeAction{
			Title: fix.Title,
			Kind:  fix.ActionKind,
			Edit: protocol.WorkspaceEdit{
				DocumentChanges: changes,
			},
			Command: fix.Command,
		}
		if pd != nil {
			action.Diagnostics = []protocol.Diagnostic{*pd}
		}
		actions = append(actions, action)
	}
	return actions, nil
}

func sameDiagnostic(pd protocol.Diagnostic, sd *source.Diagnostic) bool {
	return pd.Message == strings.TrimSpace(sd.Message) && // extra space may have been trimmed when converting to protocol.Diagnostic
		protocol.CompareRange(pd.Range, sd.Range) == 0 && pd.Source == string(sd.Source)
}

func goTest(ctx context.Context, snapshot source.Snapshot, uri span.URI, rng protocol.Range) ([]protocol.CodeAction, error) {
	fh, err := snapshot.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	fns, err := source.TestsAndBenchmarks(ctx, snapshot, fh)
	if err != nil {
		return nil, err
	}

	var tests, benchmarks []string
	for _, fn := range fns.Tests {
		if !protocol.Intersect(fn.Rng, rng) {
			continue
		}
		tests = append(tests, fn.Name)
	}
	for _, fn := range fns.Benchmarks {
		if !protocol.Intersect(fn.Rng, rng) {
			continue
		}
		benchmarks = append(benchmarks, fn.Name)
	}

	if len(tests) == 0 && len(benchmarks) == 0 {
		return nil, nil
	}

	cmd, err := command.NewTestCommand("Run tests and benchmarks", protocol.URIFromSpanURI(uri), tests, benchmarks)
	if err != nil {
		return nil, err
	}
	return []protocol.CodeAction{{
		Title:   cmd.Title,
		Kind:    protocol.GoTest,
		Command: &cmd,
	}}, nil
}
