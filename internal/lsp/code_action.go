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

	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/mod"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func (s *Server) codeAction(ctx context.Context, params *protocol.CodeActionParams) ([]protocol.CodeAction, error) {
	snapshot, fh, ok, err := s.beginFileRequest(params.TextDocument.URI, source.UnknownKind)
	if !ok {
		return nil, err
	}
	uri := fh.Identity().URI

	// Determine the supported actions for this file kind.
	supportedCodeActions, ok := snapshot.View().Options().SupportedCodeActions[fh.Identity().Kind]
	if !ok {
		return nil, fmt.Errorf("no supported code actions for %v file kind", fh.Identity().Kind)
	}

	// The Only field of the context specifies which code actions the client wants.
	// If Only is empty, assume that the client wants all of the possible code actions.
	var wanted map[protocol.CodeActionKind]bool
	if len(params.Context.Only) == 0 {
		wanted = supportedCodeActions
	} else {
		wanted = make(map[protocol.CodeActionKind]bool)
		for _, only := range params.Context.Only {
			wanted[only] = supportedCodeActions[only]
		}
	}
	if len(wanted) == 0 {
		return nil, fmt.Errorf("no supported code action to execute for %s, wanted %v", uri, params.Context.Only)
	}

	var codeActions []protocol.CodeAction
	switch fh.Identity().Kind {
	case source.Mod:
		if diagnostics := params.Context.Diagnostics; len(diagnostics) > 0 {
			modFixes, err := mod.SuggestedFixes(ctx, snapshot, fh, diagnostics)
			if err != nil {
				return nil, err
			}
			codeActions = append(codeActions, modFixes...)
		}
		if wanted[protocol.SourceOrganizeImports] {
			codeActions = append(codeActions, protocol.CodeAction{
				Title: "Tidy",
				Kind:  protocol.SourceOrganizeImports,
				Command: &protocol.Command{
					Title:     "Tidy",
					Command:   "tidy",
					Arguments: []interface{}{fh.Identity().URI},
				},
			})
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
				return nil, err
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
		// Check for context cancellation before processing analysis fixes.
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		// Retrieve any necessary analysis fixes or edits.
		if (wanted[protocol.QuickFix] || wanted[protocol.SourceFixAll]) && len(diagnostics) > 0 {
			analysisQuickFixes, highConfidenceEdits, err := analysisFixes(ctx, snapshot, fh, diagnostics)
			if err != nil {
				return nil, err
			}
			if wanted[protocol.QuickFix] {
				// Add the quick fixes reported by go/analysis.
				codeActions = append(codeActions, analysisQuickFixes...)

				// If there are any diagnostics relating to the go.mod file,
				// add their corresponding quick fixes.
				moduleQuickFixes, err := getModuleQuickFixes(ctx, snapshot, diagnostics)
				if err != nil {
					return nil, err
				}
				codeActions = append(codeActions, moduleQuickFixes...)
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
	default:
		// Unsupported file kind for a code action.
		return nil, nil
	}
	return codeActions, nil
}

var missingRequirementRe = regexp.MustCompile(`(.+) is not in your go.mod file`)

func getModuleQuickFixes(ctx context.Context, snapshot source.Snapshot, diagnostics []protocol.Diagnostic) ([]protocol.CodeAction, error) {
	// Don't bother getting quick fixes if we have no relevant diagnostics.
	var missingDeps map[string]protocol.Diagnostic
	for _, diagnostic := range diagnostics {
		matches := missingRequirementRe.FindStringSubmatch(diagnostic.Message)
		if len(matches) != 2 {
			continue
		}
		if missingDeps == nil {
			missingDeps = make(map[string]protocol.Diagnostic)
		}
		missingDeps[matches[1]] = diagnostic
	}
	if len(missingDeps) == 0 {
		return nil, nil
	}
	// Get suggested fixes for each missing dependency.
	edits, err := mod.SuggestedGoFixes(ctx, snapshot)
	if err != nil {
		return nil, err
	}
	var codeActions []protocol.CodeAction
	for dep, diagnostic := range missingDeps {
		edit, ok := edits[dep]
		if !ok {
			continue
		}
		codeActions = append(codeActions, protocol.CodeAction{
			Title:       fmt.Sprintf("Add %s to go.mod", dep),
			Diagnostics: []protocol.Diagnostic{diagnostic},
			Edit: protocol.WorkspaceEdit{
				DocumentChanges: []protocol.TextDocumentEdit{edit},
			},
			Kind: protocol.QuickFix,
		})
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

func analysisFixes(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle, diagnostics []protocol.Diagnostic) ([]protocol.CodeAction, []protocol.TextDocumentEdit, error) {
	if len(diagnostics) == 0 {
		return nil, nil, nil
	}

	var codeActions []protocol.CodeAction
	var sourceFixAllEdits []protocol.TextDocumentEdit

	phs, err := snapshot.PackageHandles(ctx, fh)
	if err != nil {
		return nil, nil, err
	}
	// We get the package that source.Diagnostics would've used. This is hack.
	// TODO(golang/go#32443): The correct solution will be to cache diagnostics per-file per-snapshot.
	ph, err := source.WidestPackageHandle(phs)
	if err != nil {
		return nil, nil, err
	}
	for _, diag := range diagnostics {
		srcErr, analyzer, ok := findSourceError(ctx, snapshot, ph.ID(), diag)
		if !ok {
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
				fh, err := snapshot.GetFile(uri)
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
	var analyzer *source.Analyzer

	// If the source is "compiler", we expect a type error analyzer.
	if diag.Source == "compiler" {
		for _, a := range snapshot.View().Options().TypeErrorAnalyzers {
			if a.FixesError(diag.Message) {
				analyzer = &a
				break
			}
		}
	} else {
		// This code assumes that the analyzer name is the Source of the diagnostic.
		// If this ever changes, this will need to be addressed.
		if a, ok := snapshot.View().Options().DefaultAnalyzers[diag.Source]; ok {
			analyzer = &a
		}
	}
	if analyzer == nil || !analyzer.Enabled(snapshot) {
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

func documentChanges(fh source.FileHandle, edits []protocol.TextEdit) []protocol.TextDocumentEdit {
	return []protocol.TextDocumentEdit{
		{
			TextDocument: protocol.VersionedTextDocumentIdentifier{
				Version: fh.Identity().Version,
				TextDocumentIdentifier: protocol.TextDocumentIdentifier{
					URI: protocol.URIFromSpanURI(fh.Identity().URI),
				},
			},
			Edits: edits,
		},
	}
}
