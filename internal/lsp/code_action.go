// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/mod"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (s *Server) codeAction(ctx context.Context, params *protocol.CodeActionParams) ([]protocol.CodeAction, error) {
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	defer release()
	if !ok {
		return nil, err
	}
	uri := fh.URI()

	// Determine the supported actions for this file kind.
	supportedCodeActions, ok := snapshot.View().Options().SupportedCodeActions[fh.Kind()]
	if !ok {
		return nil, fmt.Errorf("no supported code actions for %v file kind", fh.Kind())
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
			wanted[only] = supportedCodeActions[only] || explicit[only]
		}
	}
	if len(wanted) == 0 {
		return nil, fmt.Errorf("no supported code action to execute for %s, wanted %v", uri, params.Context.Only)
	}

	var codeActions []protocol.CodeAction
	switch fh.Kind() {
	case source.Mod:
		if diagnostics := params.Context.Diagnostics; len(diagnostics) > 0 {
			modQuickFixes, err := moduleQuickFixes(ctx, snapshot, fh, diagnostics)
			if source.IsNonFatalGoModError(err) {
				return nil, nil
			}
			if err != nil {
				return nil, err
			}
			codeActions = append(codeActions, modQuickFixes...)
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

			// Fix unresolved imports with "go get". This is separate from the
			// goimports fixes because goimports will not remove an import
			// that appears to be used, even if currently unresolved.
			actions, err := goGetFixes(ctx, snapshot, fh.URI(), diagnostics)
			if err != nil {
				return nil, err
			}
			codeActions = append(codeActions, actions...)

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
		if (wanted[protocol.QuickFix] || wanted[protocol.SourceFixAll]) && len(diagnostics) > 0 {
			analysisQuickFixes, highConfidenceEdits, err := analysisFixes(ctx, snapshot, pkg, diagnostics)
			if err != nil {
				return nil, err
			}
			if wanted[protocol.QuickFix] {
				// Add the quick fixes reported by go/analysis.
				codeActions = append(codeActions, analysisQuickFixes...)

				// If there are any diagnostics relating to the go.mod file,
				// add their corresponding quick fixes.
				modQuickFixes, err := moduleQuickFixes(ctx, snapshot, fh, diagnostics)
				if source.IsNonFatalGoModError(err) {
					// Not a fatal error.
					event.Error(ctx, "module suggested fixes failed", err, tag.Directory.Of(snapshot.View().Folder()))
				}
				codeActions = append(codeActions, modQuickFixes...)
			}
			if wanted[protocol.SourceFixAll] && len(highConfidenceEdits) > 0 {
				codeActions = append(codeActions, protocol.CodeAction{
					Title: "Simplifications",
					Kind:  protocol.SourceFixAll,
					Edit: protocol.WorkspaceEdit{
						DocumentChanges: highConfidenceEdits,
					},
				})
			}
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		// Add any suggestions that do not necessarily fix any diagnostics.
		if wanted[protocol.RefactorRewrite] {
			fixes, err := convenienceFixes(ctx, snapshot, pkg, uri, params.Range)
			if err != nil {
				return nil, err
			}
			codeActions = append(codeActions, fixes...)
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
	return codeActions, nil
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

func analysisFixes(ctx context.Context, snapshot source.Snapshot, pkg source.Package, diagnostics []protocol.Diagnostic) ([]protocol.CodeAction, []protocol.TextDocumentEdit, error) {
	if len(diagnostics) == 0 {
		return nil, nil, nil
	}
	var (
		codeActions       []protocol.CodeAction
		sourceFixAllEdits []protocol.TextDocumentEdit
	)
	for _, diag := range diagnostics {
		srcErr, analyzer, ok := findSourceError(ctx, snapshot, pkg.ID(), diag)
		if !ok {
			continue
		}
		// If the suggested fix for the diagnostic is expected to be separate,
		// see if there are any supported commands available.
		if analyzer.Command != nil {
			action, err := diagnosticToCommandCodeAction(ctx, snapshot, srcErr, &diag, protocol.QuickFix)
			if err != nil {
				return nil, nil, err
			}
			codeActions = append(codeActions, *action)
			continue
		}
		for _, fix := range srcErr.SuggestedFixes {
			action := protocol.CodeAction{
				Title:       fix.Title,
				Kind:        protocol.QuickFix,
				Diagnostics: []protocol.Diagnostic{diag},
				Edit:        protocol.WorkspaceEdit{},
			}
			for uri, edits := range fix.Edits {
				fh, err := snapshot.GetVersionedFile(ctx, uri)
				if err != nil {
					return nil, nil, err
				}
				docChanges := documentChanges(fh, edits)
				if analyzer.HighConfidence {
					sourceFixAllEdits = append(sourceFixAllEdits, docChanges...)
				}
				action.Edit.DocumentChanges = append(action.Edit.DocumentChanges, docChanges...)
			}
			codeActions = append(codeActions, action)
		}
	}
	return codeActions, sourceFixAllEdits, nil
}

func findSourceError(ctx context.Context, snapshot source.Snapshot, pkgID string, diag protocol.Diagnostic) (*source.Error, source.Analyzer, bool) {
	analyzer := diagnosticToAnalyzer(snapshot, diag.Source, diag.Message)
	if analyzer == nil {
		return nil, source.Analyzer{}, false
	}
	analysisErrors, err := snapshot.Analyze(ctx, pkgID, analyzer.Analyzer)
	if err != nil {
		return nil, source.Analyzer{}, false
	}
	for _, err := range analysisErrors {
		if err.Message != diag.Message {
			continue
		}
		if protocol.CompareRange(err.Range, diag.Range) != 0 {
			continue
		}
		if err.Category != analyzer.Analyzer.Name {
			continue
		}
		// The error matches.
		return err, *analyzer, true
	}
	return nil, source.Analyzer{}, false
}

// diagnosticToAnalyzer return the analyzer associated with a given diagnostic.
// It assumes that the diagnostic's source will be the name of the analyzer.
// If this changes, this approach will need to be reworked.
func diagnosticToAnalyzer(snapshot source.Snapshot, src, msg string) (analyzer *source.Analyzer) {
	// Make sure that the analyzer we found is enabled.
	defer func() {
		if analyzer != nil && !analyzer.IsEnabled(snapshot.View()) {
			analyzer = nil
		}
	}()
	if a, ok := snapshot.View().Options().DefaultAnalyzers[src]; ok {
		return &a
	}
	if a, ok := snapshot.View().Options().StaticcheckAnalyzers[src]; ok {
		return &a
	}
	if a, ok := snapshot.View().Options().ConvenienceAnalyzers[src]; ok {
		return &a
	}
	// Hack: We publish diagnostics with the source "compiler" for type errors,
	// but these analyzers have different names. Try both possibilities.
	if a, ok := snapshot.View().Options().TypeErrorAnalyzers[src]; ok {
		return &a
	}
	if src != "compiler" {
		return nil
	}
	for _, a := range snapshot.View().Options().TypeErrorAnalyzers {
		if a.FixesError(msg) {
			return &a
		}
	}
	return nil
}

var importErrorRe = regexp.MustCompile(`could not import ([^\s]+)`)

func goGetFixes(ctx context.Context, snapshot source.Snapshot, uri span.URI, diagnostics []protocol.Diagnostic) ([]protocol.CodeAction, error) {
	if snapshot.GoModForFile(ctx, uri) == "" {
		// Go get only supports module mode for now.
		return nil, nil
	}

	var actions []protocol.CodeAction
	for _, diag := range diagnostics {
		matches := importErrorRe.FindStringSubmatch(diag.Message)
		if len(matches) == 0 {
			return nil, nil
		}
		args, err := source.MarshalArgs(uri, matches[1])
		if err != nil {
			return nil, err
		}
		actions = append(actions, protocol.CodeAction{
			Title:       fmt.Sprintf("go get package %v", matches[1]),
			Diagnostics: []protocol.Diagnostic{diag},
			Kind:        protocol.QuickFix,
			Command: &protocol.Command{
				Title:     source.CommandGoGetPackage.Title,
				Command:   source.CommandGoGetPackage.ID(),
				Arguments: args,
			},
		})
	}
	return actions, nil
}

func convenienceFixes(ctx context.Context, snapshot source.Snapshot, pkg source.Package, uri span.URI, rng protocol.Range) ([]protocol.CodeAction, error) {
	var analyzers []*analysis.Analyzer
	for _, a := range snapshot.View().Options().ConvenienceAnalyzers {
		if !a.IsEnabled(snapshot.View()) {
			continue
		}
		if a.Command == nil {
			event.Error(ctx, "convenienceFixes", fmt.Errorf("no suggested fixes for convenience analyzer %s", a.Analyzer.Name))
			continue
		}
		analyzers = append(analyzers, a.Analyzer)
	}
	diagnostics, err := snapshot.Analyze(ctx, pkg.ID(), analyzers...)
	if err != nil {
		return nil, err
	}
	var codeActions []protocol.CodeAction
	for _, d := range diagnostics {
		// For now, only show diagnostics for matching lines. Maybe we should
		// alter this behavior in the future, depending on the user experience.
		if d.URI != uri {
			continue
		}

		if !protocol.Intersect(d.Range, rng) {
			continue
		}
		action, err := diagnosticToCommandCodeAction(ctx, snapshot, d, nil, protocol.RefactorRewrite)
		if err != nil {
			return nil, err
		}
		codeActions = append(codeActions, *action)
	}
	return codeActions, nil
}

func diagnosticToCommandCodeAction(ctx context.Context, snapshot source.Snapshot, e *source.Error, d *protocol.Diagnostic, kind protocol.CodeActionKind) (*protocol.CodeAction, error) {
	// The fix depends on the category of the analyzer. The diagnostic may be
	// nil, so use the error's category.
	analyzer := diagnosticToAnalyzer(snapshot, e.Category, e.Message)
	if analyzer == nil {
		return nil, fmt.Errorf("no convenience analyzer for category %s", e.Category)
	}
	if analyzer.Command == nil {
		return nil, fmt.Errorf("no command for convenience analyzer %s", analyzer.Analyzer.Name)
	}
	jsonArgs, err := source.MarshalArgs(e.URI, e.Range)
	if err != nil {
		return nil, err
	}
	var diagnostics []protocol.Diagnostic
	if d != nil {
		diagnostics = append(diagnostics, *d)
	}
	return &protocol.CodeAction{
		Title:       e.Message,
		Kind:        kind,
		Diagnostics: diagnostics,
		Command: &protocol.Command{
			Command:   analyzer.Command.ID(),
			Title:     e.Message,
			Arguments: jsonArgs,
		},
	}, nil
}

func extractionFixes(ctx context.Context, snapshot source.Snapshot, pkg source.Package, uri span.URI, rng protocol.Range) ([]protocol.CodeAction, error) {
	if rng.Start == rng.End {
		return nil, nil
	}
	fh, err := snapshot.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	jsonArgs, err := source.MarshalArgs(uri, rng)
	if err != nil {
		return nil, err
	}
	var actions []protocol.CodeAction
	for _, command := range []*source.Command{
		source.CommandExtractFunction,
		source.CommandExtractVariable,
	} {
		if !command.Applies(ctx, snapshot, fh, rng) {
			continue
		}
		actions = append(actions, protocol.CodeAction{
			Title: command.Title,
			Kind:  protocol.RefactorExtract,
			Command: &protocol.Command{
				Command:   command.ID(),
				Arguments: jsonArgs,
			},
		})
	}
	return actions, nil
}

func documentChanges(fh source.VersionedFileHandle, edits []protocol.TextEdit) []protocol.TextDocumentEdit {
	return []protocol.TextDocumentEdit{
		{
			TextDocument: protocol.VersionedTextDocumentIdentifier{
				Version: fh.Version(),
				TextDocumentIdentifier: protocol.TextDocumentIdentifier{
					URI: protocol.URIFromSpanURI(fh.URI()),
				},
			},
			Edits: edits,
		},
	}
}

func moduleQuickFixes(ctx context.Context, snapshot source.Snapshot, fh source.VersionedFileHandle, diagnostics []protocol.Diagnostic) ([]protocol.CodeAction, error) {
	var modFH source.VersionedFileHandle
	switch fh.Kind() {
	case source.Mod:
		modFH = fh
	case source.Go:
		modURI := snapshot.GoModForFile(ctx, fh.URI())
		if modURI == "" {
			return nil, nil
		}
		var err error
		modFH, err = snapshot.GetVersionedFile(ctx, modURI)
		if err != nil {
			return nil, err
		}
	}
	errors, err := mod.ErrorsForMod(ctx, snapshot, modFH)
	if err != nil {
		return nil, err
	}
	var quickFixes []protocol.CodeAction
	for _, e := range errors {
		var diag *protocol.Diagnostic
		for _, d := range diagnostics {
			if sameDiagnostic(d, e) {
				diag = &d
				break
			}
		}
		if diag == nil {
			continue
		}
		for _, fix := range e.SuggestedFixes {
			action := protocol.CodeAction{
				Title:       fix.Title,
				Kind:        protocol.QuickFix,
				Diagnostics: []protocol.Diagnostic{*diag},
				Edit:        protocol.WorkspaceEdit{},
				Command:     fix.Command,
			}
			for uri, edits := range fix.Edits {
				if uri != modFH.URI() {
					continue
				}
				action.Edit.DocumentChanges = append(action.Edit.DocumentChanges, protocol.TextDocumentEdit{
					TextDocument: protocol.VersionedTextDocumentIdentifier{
						Version: modFH.Version(),
						TextDocumentIdentifier: protocol.TextDocumentIdentifier{
							URI: protocol.URIFromSpanURI(modFH.URI()),
						},
					},
					Edits: edits,
				})
			}
			quickFixes = append(quickFixes, action)
		}
	}
	return quickFixes, nil
}

func sameDiagnostic(d protocol.Diagnostic, e *source.Error) bool {
	return d.Message == e.Message && protocol.CompareRange(d.Range, e.Range) == 0 && d.Source == e.Category
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

	jsonArgs, err := source.MarshalArgs(uri, tests, benchmarks)
	if err != nil {
		return nil, err
	}
	return []protocol.CodeAction{{
		Title: source.CommandTest.Name,
		Kind:  protocol.GoTest,
		Command: &protocol.Command{
			Title:     source.CommandTest.Title,
			Command:   source.CommandTest.ID(),
			Arguments: jsonArgs,
		},
	}}, nil
}
