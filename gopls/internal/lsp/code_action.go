// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"go/ast"
	"sort"
	"strings"

	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/analysis/fillstruct"
	"golang.org/x/tools/gopls/internal/lsp/analysis/infertypeargs"
	"golang.org/x/tools/gopls/internal/lsp/analysis/stubmethods"
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
	ctx, done := event.Start(ctx, "lsp.Server.codeAction")
	defer done()

	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	defer release()
	if !ok {
		return nil, err
	}
	uri := fh.URI()

	// Determine the supported actions for this file kind.
	kind := snapshot.FileKind(fh)
	supportedCodeActions, ok := snapshot.Options().SupportedCodeActions[kind]
	if !ok {
		return nil, fmt.Errorf("no supported code actions for %v file kind", kind)
	}
	if len(supportedCodeActions) == 0 {
		return nil, nil // not an error if there are none supported
	}

	// The Only field of the context specifies which code actions the client wants.
	// If Only is empty, assume that the client wants all of the non-explicit code actions.
	var want map[protocol.CodeActionKind]bool
	{
		// Explicit Code Actions are opt-in and shouldn't be returned to the client unless
		// requested using Only.
		// TODO: Add other CodeLenses such as GoGenerate, RegenerateCgo, etc..
		explicit := map[protocol.CodeActionKind]bool{
			protocol.GoTest: true,
		}

		if len(params.Context.Only) == 0 {
			want = supportedCodeActions
		} else {
			want = make(map[protocol.CodeActionKind]bool)
			for _, only := range params.Context.Only {
				for k, v := range supportedCodeActions {
					if only == k || strings.HasPrefix(string(k), string(only)+".") {
						want[k] = want[k] || v
					}
				}
				want[only] = want[only] || explicit[only]
			}
		}
	}
	if len(want) == 0 {
		return nil, fmt.Errorf("no supported code action to execute for %s, wanted %v", uri, params.Context.Only)
	}

	switch kind {
	case source.Mod:
		var actions []protocol.CodeAction

		fixes, err := s.codeActionsMatchingDiagnostics(ctx, fh.URI(), snapshot, params.Context.Diagnostics, want)
		if err != nil {
			return nil, err
		}

		// Group vulnerability fixes by their range, and select only the most
		// appropriate upgrades.
		//
		// TODO(rfindley): can this instead be accomplished on the diagnosis side,
		// so that code action handling remains uniform?
		vulnFixes := make(map[protocol.Range][]protocol.CodeAction)
	searchFixes:
		for _, fix := range fixes {
			for _, diag := range fix.Diagnostics {
				if diag.Source == string(source.Govulncheck) || diag.Source == string(source.Vulncheck) {
					vulnFixes[diag.Range] = append(vulnFixes[diag.Range], fix)
					continue searchFixes
				}
			}
			actions = append(actions, fix)
		}

		for _, fixes := range vulnFixes {
			fixes = mod.SelectUpgradeCodeActions(fixes)
			actions = append(actions, fixes...)
		}

		return actions, nil

	case source.Go:
		diagnostics := params.Context.Diagnostics

		// Don't suggest fixes for generated files, since they are generally
		// not useful and some editors may apply them automatically on save.
		if source.IsGenerated(ctx, snapshot, uri) {
			return nil, nil
		}

		actions, err := s.codeActionsMatchingDiagnostics(ctx, uri, snapshot, diagnostics, want)
		if err != nil {
			return nil, err
		}

		// Only compute quick fixes if there are any diagnostics to fix.
		wantQuickFixes := want[protocol.QuickFix] && len(diagnostics) > 0

		// Code actions requiring syntax information alone.
		if wantQuickFixes || want[protocol.SourceOrganizeImports] || want[protocol.RefactorExtract] {
			pgf, err := snapshot.ParseGo(ctx, fh, source.ParseFull)
			if err != nil {
				return nil, err
			}

			// Process any missing imports and pair them with the diagnostics they
			// fix.
			if wantQuickFixes || want[protocol.SourceOrganizeImports] {
				importEdits, importEditsPerFix, err := source.AllImportsFixes(ctx, snapshot, pgf)
				if err != nil {
					event.Error(ctx, "imports fixes", err, tag.File.Of(fh.URI().Filename()))
					importEdits = nil
					importEditsPerFix = nil
				}

				// Separate this into a set of codeActions per diagnostic, where
				// each action is the addition, removal, or renaming of one import.
				if wantQuickFixes {
					for _, importFix := range importEditsPerFix {
						fixed := fixedByImportFix(importFix.Fix, diagnostics)
						if len(fixed) == 0 {
							continue
						}
						actions = append(actions, protocol.CodeAction{
							Title: importFixTitle(importFix.Fix),
							Kind:  protocol.QuickFix,
							Edit: &protocol.WorkspaceEdit{
								DocumentChanges: documentChanges(fh, importFix.Edits),
							},
							Diagnostics: fixed,
						})
					}
				}

				// Send all of the import edits as one code action if the file is
				// being organized.
				if want[protocol.SourceOrganizeImports] && len(importEdits) > 0 {
					actions = append(actions, protocol.CodeAction{
						Title: "Organize Imports",
						Kind:  protocol.SourceOrganizeImports,
						Edit: &protocol.WorkspaceEdit{
							DocumentChanges: documentChanges(fh, importEdits),
						},
					})
				}
			}

			if want[protocol.RefactorExtract] {
				extractions, err := refactorExtract(ctx, snapshot, pgf, params.Range)
				if err != nil {
					return nil, err
				}
				actions = append(actions, extractions...)
			}
		}

		var stubMethodsDiagnostics []protocol.Diagnostic
		if wantQuickFixes && snapshot.Options().IsAnalyzerEnabled(stubmethods.Analyzer.Name) {
			for _, pd := range diagnostics {
				if stubmethods.MatchesMessage(pd.Message) {
					stubMethodsDiagnostics = append(stubMethodsDiagnostics, pd)
				}
			}
		}

		// Code actions requiring type information.
		if len(stubMethodsDiagnostics) > 0 ||
			want[protocol.RefactorRewrite] ||
			want[protocol.RefactorInline] ||
			want[protocol.GoTest] {
			pkg, pgf, err := source.NarrowestPackageForFile(ctx, snapshot, fh.URI())
			if err != nil {
				return nil, err
			}
			for _, pd := range stubMethodsDiagnostics {
				start, end, err := pgf.RangePos(pd.Range)
				if err != nil {
					return nil, err
				}
				action, ok, err := func() (_ protocol.CodeAction, _ bool, rerr error) {
					// golang/go#61693: code actions were refactored to run outside of the
					// analysis framework, but as a result they lost their panic recovery.
					//
					// Stubmethods "should never fail"", but put back the panic recovery as a
					// defensive measure.
					defer func() {
						if r := recover(); r != nil {
							rerr = bug.Errorf("stubmethods panicked: %v", r)
						}
					}()
					d, ok := stubmethods.DiagnosticForError(pkg.FileSet(), pgf.File, start, end, pd.Message, pkg.GetTypesInfo())
					if !ok {
						return protocol.CodeAction{}, false, nil
					}
					cmd, err := command.NewApplyFixCommand(d.Message, command.ApplyFixArgs{
						URI:   protocol.URIFromSpanURI(pgf.URI),
						Fix:   source.StubMethods,
						Range: pd.Range,
					})
					if err != nil {
						return protocol.CodeAction{}, false, err
					}
					return protocol.CodeAction{
						Title:       d.Message,
						Kind:        protocol.QuickFix,
						Command:     &cmd,
						Diagnostics: []protocol.Diagnostic{pd},
					}, true, nil
				}()
				if err != nil {
					return nil, err
				}
				if ok {
					actions = append(actions, action)
				}
			}

			if want[protocol.RefactorRewrite] {
				rewrites, err := refactorRewrite(ctx, snapshot, pkg, pgf, fh, params.Range)
				if err != nil {
					return nil, err
				}
				actions = append(actions, rewrites...)
			}

			if want[protocol.RefactorInline] {
				rewrites, err := refactorInline(ctx, snapshot, pkg, pgf, fh, params.Range)
				if err != nil {
					return nil, err
				}
				actions = append(actions, rewrites...)
			}

			if want[protocol.GoTest] {
				fixes, err := goTest(ctx, snapshot, pkg, pgf, params.Range)
				if err != nil {
					return nil, err
				}
				actions = append(actions, fixes...)
			}
		}

		return actions, nil

	default:
		// Unsupported file kind for a code action.
		return nil, nil
	}
}

func (s *Server) findMatchingDiagnostics(uri span.URI, pd protocol.Diagnostic) []*source.Diagnostic {
	s.diagnosticsMu.Lock()
	defer s.diagnosticsMu.Unlock()

	var sds []*source.Diagnostic
	for _, report := range s.diagnostics[uri].reports {
		for _, sd := range report.diags {
			sameDiagnostic := (pd.Message == strings.TrimSpace(sd.Message) && // extra space may have been trimmed when converting to protocol.Diagnostic
				protocol.CompareRange(pd.Range, sd.Range) == 0 &&
				pd.Source == string(sd.Source))

			if sameDiagnostic {
				sds = append(sds, sd)
			}
		}
	}
	return sds
}

func (s *Server) getSupportedCodeActions() []protocol.CodeActionKind {
	allCodeActionKinds := make(map[protocol.CodeActionKind]struct{})
	for _, kinds := range s.Options().SupportedCodeActions {
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

// fixedByImportFix filters the provided slice of diagnostics to those that
// would be fixed by the provided imports fix.
func fixedByImportFix(fix *imports.ImportFix, diagnostics []protocol.Diagnostic) []protocol.Diagnostic {
	var results []protocol.Diagnostic
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

func refactorExtract(ctx context.Context, snapshot source.Snapshot, pgf *source.ParsedGoFile, rng protocol.Range) ([]protocol.CodeAction, error) {
	if rng.Start == rng.End {
		return nil, nil
	}

	start, end, err := pgf.RangePos(rng)
	if err != nil {
		return nil, err
	}
	puri := protocol.URIFromSpanURI(pgf.URI)
	var commands []protocol.Command
	if _, ok, methodOk, _ := source.CanExtractFunction(pgf.Tok, start, end, pgf.Src, pgf.File); ok {
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
	if _, _, ok, _ := source.CanExtractVariable(start, end, pgf.File); ok {
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

func refactorRewrite(ctx context.Context, snapshot source.Snapshot, pkg source.Package, pgf *source.ParsedGoFile, fh source.FileHandle, rng protocol.Range) (_ []protocol.CodeAction, rerr error) {
	// golang/go#61693: code actions were refactored to run outside of the
	// analysis framework, but as a result they lost their panic recovery.
	//
	// These code actions should never fail, but put back the panic recovery as a
	// defensive measure.
	defer func() {
		if r := recover(); r != nil {
			rerr = bug.Errorf("refactor.rewrite code actions panicked: %v", r)
		}
	}()

	var actions []protocol.CodeAction

	if canRemoveParameter(pkg, pgf, rng) {
		cmd, err := command.NewChangeSignatureCommand("remove unused parameter", command.ChangeSignatureArgs{
			RemoveParameter: protocol.Location{
				URI:   protocol.URIFromSpanURI(pgf.URI),
				Range: rng,
			},
		})
		if err != nil {
			return nil, err
		}
		actions = append(actions, protocol.CodeAction{
			Title:   "Refactor: remove unused parameter",
			Kind:    protocol.RefactorRewrite,
			Command: &cmd,
		})
	}

	start, end, err := pgf.RangePos(rng)
	if err != nil {
		return nil, err
	}

	var commands []protocol.Command
	if _, ok, _ := source.CanInvertIfCondition(pgf.File, start, end); ok {
		cmd, err := command.NewApplyFixCommand("Invert if condition", command.ApplyFixArgs{
			URI:   protocol.URIFromSpanURI(pgf.URI),
			Fix:   source.InvertIfCondition,
			Range: rng,
		})
		if err != nil {
			return nil, err
		}
		commands = append(commands, cmd)
	}

	// N.B.: an inspector only pays for itself after ~5 passes, which means we're
	// currently not getting a good deal on this inspection.
	//
	// TODO: Consider removing the inspection after convenienceAnalyzers are removed.
	inspect := inspector.New([]*ast.File{pgf.File})
	if snapshot.Options().IsAnalyzerEnabled(fillstruct.Analyzer.Name) {
		for _, d := range fillstruct.DiagnoseFillableStructs(inspect, start, end, pkg.GetTypes(), pkg.GetTypesInfo()) {
			rng, err := pgf.Mapper.PosRange(pgf.Tok, d.Pos, d.End)
			if err != nil {
				return nil, err
			}
			cmd, err := command.NewApplyFixCommand(d.Message, command.ApplyFixArgs{
				URI:   protocol.URIFromSpanURI(pgf.URI),
				Fix:   source.FillStruct,
				Range: rng,
			})
			if err != nil {
				return nil, err
			}
			commands = append(commands, cmd)
		}
	}

	for i := range commands {
		actions = append(actions, protocol.CodeAction{
			Title:   commands[i].Title,
			Kind:    protocol.RefactorRewrite,
			Command: &commands[i],
		})
	}

	if snapshot.Options().IsAnalyzerEnabled(infertypeargs.Analyzer.Name) {
		for _, d := range infertypeargs.DiagnoseInferableTypeArgs(pkg.FileSet(), inspect, start, end, pkg.GetTypes(), pkg.GetTypesInfo()) {
			if len(d.SuggestedFixes) != 1 {
				panic(fmt.Sprintf("unexpected number of suggested fixes from infertypeargs: %v", len(d.SuggestedFixes)))
			}
			fix := d.SuggestedFixes[0]
			var edits []protocol.TextEdit
			for _, analysisEdit := range fix.TextEdits {
				rng, err := pgf.Mapper.PosRange(pgf.Tok, analysisEdit.Pos, analysisEdit.End)
				if err != nil {
					return nil, err
				}
				edits = append(edits, protocol.TextEdit{
					Range:   rng,
					NewText: string(analysisEdit.NewText),
				})
			}
			actions = append(actions, protocol.CodeAction{
				Title: "Simplify type arguments",
				Kind:  protocol.RefactorRewrite,
				Edit: &protocol.WorkspaceEdit{
					DocumentChanges: documentChanges(fh, edits),
				},
			})
		}
	}

	return actions, nil
}

// canRemoveParameter reports whether we can remove the function parameter
// indicated by the given [start, end) range.
//
// This is true if:
//   - [start, end) is contained within an unused field or parameter name
//   - ... of a non-method function declaration.
func canRemoveParameter(pkg source.Package, pgf *source.ParsedGoFile, rng protocol.Range) bool {
	info := source.FindParam(pgf, rng)
	if info.Decl == nil || info.Field == nil {
		return false
	}

	if len(info.Field.Names) == 0 {
		return true // no names => field is unused
	}
	if info.Name == nil {
		return false // no name is indicated
	}
	if info.Name.Name == "_" {
		return true // trivially unused
	}

	obj := pkg.GetTypesInfo().Defs[info.Name]
	if obj == nil {
		return false // something went wrong
	}

	used := false
	ast.Inspect(info.Decl.Body, func(node ast.Node) bool {
		if n, ok := node.(*ast.Ident); ok && pkg.GetTypesInfo().Uses[n] == obj {
			used = true
		}
		return !used // keep going until we find a use
	})
	return !used
}

// refactorInline returns inline actions available at the specified range.
func refactorInline(ctx context.Context, snapshot source.Snapshot, pkg source.Package, pgf *source.ParsedGoFile, fh source.FileHandle, rng protocol.Range) ([]protocol.CodeAction, error) {
	var commands []protocol.Command

	// If range is within call expression, offer inline action.
	if _, fn, err := source.EnclosingStaticCall(pkg, pgf, rng); err == nil {
		cmd, err := command.NewApplyFixCommand(fmt.Sprintf("Inline call to %s", fn.Name()), command.ApplyFixArgs{
			URI:   protocol.URIFromSpanURI(pgf.URI),
			Fix:   source.InlineCall,
			Range: rng,
		})
		if err != nil {
			return nil, err
		}
		commands = append(commands, cmd)
	}

	// Convert commands to actions.
	var actions []protocol.CodeAction
	for i := range commands {
		actions = append(actions, protocol.CodeAction{
			Title:   commands[i].Title,
			Kind:    protocol.RefactorInline,
			Command: &commands[i],
		})
	}
	return actions, nil
}

func documentChanges(fh source.FileHandle, edits []protocol.TextEdit) []protocol.DocumentChanges {
	return []protocol.DocumentChanges{
		{
			TextDocumentEdit: &protocol.TextDocumentEdit{
				TextDocument: protocol.OptionalVersionedTextDocumentIdentifier{
					Version: fh.Version(),
					TextDocumentIdentifier: protocol.TextDocumentIdentifier{
						URI: protocol.URIFromSpanURI(fh.URI()),
					},
				},
				Edits: nonNilSliceTextEdit(edits),
			},
		},
	}
}

// codeActionsMatchingDiagnostics fetches code actions for the provided
// diagnostics, by first attempting to unmarshal code actions directly from the
// bundled protocol.Diagnostic.Data field, and failing that by falling back on
// fetching a matching source.Diagnostic from the set of stored diagnostics for
// this file.
func (s *Server) codeActionsMatchingDiagnostics(ctx context.Context, uri span.URI, snapshot source.Snapshot, pds []protocol.Diagnostic, want map[protocol.CodeActionKind]bool) ([]protocol.CodeAction, error) {
	var actions []protocol.CodeAction
	var unbundled []protocol.Diagnostic // diagnostics without bundled code actions in their Data field
	for _, pd := range pds {
		bundled := source.BundledQuickFixes(pd)
		if len(bundled) > 0 {
			for _, fix := range bundled {
				if want[fix.Kind] {
					actions = append(actions, fix)
				}
			}
		} else {
			// No bundled actions: keep searching for a match.
			unbundled = append(unbundled, pd)
		}
	}

	for _, pd := range unbundled {
		for _, sd := range s.findMatchingDiagnostics(uri, pd) {
			diagActions, err := codeActionsForDiagnostic(ctx, snapshot, sd, &pd, want)
			if err != nil {
				return nil, err
			}
			actions = append(actions, diagActions...)
		}
	}
	return actions, nil
}

func codeActionsForDiagnostic(ctx context.Context, snapshot source.Snapshot, sd *source.Diagnostic, pd *protocol.Diagnostic, want map[protocol.CodeActionKind]bool) ([]protocol.CodeAction, error) {
	var actions []protocol.CodeAction
	for _, fix := range sd.SuggestedFixes {
		if !want[fix.ActionKind] {
			continue
		}
		changes := []protocol.DocumentChanges{} // must be a slice
		for uri, edits := range fix.Edits {
			fh, err := snapshot.ReadFile(ctx, uri)
			if err != nil {
				return nil, err
			}
			changes = append(changes, documentChanges(fh, edits)...)
		}
		action := protocol.CodeAction{
			Title: fix.Title,
			Kind:  fix.ActionKind,
			Edit: &protocol.WorkspaceEdit{
				DocumentChanges: changes,
			},
			Command: fix.Command,
		}
		action.Diagnostics = []protocol.Diagnostic{*pd}
		actions = append(actions, action)
	}
	return actions, nil
}

func goTest(ctx context.Context, snapshot source.Snapshot, pkg source.Package, pgf *source.ParsedGoFile, rng protocol.Range) ([]protocol.CodeAction, error) {
	fns, err := source.TestsAndBenchmarks(ctx, snapshot, pkg, pgf)
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

	cmd, err := command.NewTestCommand("Run tests and benchmarks", protocol.URIFromSpanURI(pgf.URI), tests, benchmarks)
	if err != nil {
		return nil, err
	}
	return []protocol.CodeAction{{
		Title:   cmd.Title,
		Kind:    protocol.GoTest,
		Command: &cmd,
	}}, nil
}

type unit = struct{}
