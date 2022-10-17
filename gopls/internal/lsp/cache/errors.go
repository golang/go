// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"fmt"
	"go/scanner"
	"go/token"
	"go/types"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/typesinternal"
)

func goPackagesErrorDiagnostics(snapshot *snapshot, pkg *pkg, e packages.Error) ([]*source.Diagnostic, error) {
	if msg, spn, ok := parseGoListImportCycleError(snapshot, e, pkg); ok {
		rng, err := spanToRange(pkg, spn)
		if err != nil {
			return nil, err
		}
		return []*source.Diagnostic{{
			URI:      spn.URI(),
			Range:    rng,
			Severity: protocol.SeverityError,
			Source:   source.ListError,
			Message:  msg,
		}}, nil
	}

	var spn span.Span
	if e.Pos == "" {
		spn = parseGoListError(e.Msg, pkg.m.Config.Dir)
		// We may not have been able to parse a valid span. Apply the errors to all files.
		if _, err := spanToRange(pkg, spn); err != nil {
			var diags []*source.Diagnostic
			for _, cgf := range pkg.compiledGoFiles {
				diags = append(diags, &source.Diagnostic{
					URI:      cgf.URI,
					Severity: protocol.SeverityError,
					Source:   source.ListError,
					Message:  e.Msg,
				})
			}
			return diags, nil
		}
	} else {
		spn = span.ParseInDir(e.Pos, pkg.m.Config.Dir)
	}

	rng, err := spanToRange(pkg, spn)
	if err != nil {
		return nil, err
	}
	return []*source.Diagnostic{{
		URI:      spn.URI(),
		Range:    rng,
		Severity: protocol.SeverityError,
		Source:   source.ListError,
		Message:  e.Msg,
	}}, nil
}

func parseErrorDiagnostics(snapshot *snapshot, pkg *pkg, errList scanner.ErrorList) ([]*source.Diagnostic, error) {
	// The first parser error is likely the root cause of the problem.
	if errList.Len() <= 0 {
		return nil, fmt.Errorf("no errors in %v", errList)
	}
	e := errList[0]
	pgf, err := pkg.File(span.URIFromPath(e.Pos.Filename))
	if err != nil {
		return nil, err
	}
	pos := pgf.Tok.Pos(e.Pos.Offset)
	spn, err := span.NewRange(pgf.Tok, pos, pos).Span()
	if err != nil {
		return nil, err
	}
	rng, err := spanToRange(pkg, spn)
	if err != nil {
		return nil, err
	}
	return []*source.Diagnostic{{
		URI:      spn.URI(),
		Range:    rng,
		Severity: protocol.SeverityError,
		Source:   source.ParseError,
		Message:  e.Msg,
	}}, nil
}

var importErrorRe = regexp.MustCompile(`could not import ([^\s]+)`)
var unsupportedFeatureRe = regexp.MustCompile(`.*require.* go(\d+\.\d+) or later`)

func typeErrorDiagnostics(snapshot *snapshot, pkg *pkg, e extendedError) ([]*source.Diagnostic, error) {
	code, spn, err := typeErrorData(snapshot.FileSet(), pkg, e.primary)
	if err != nil {
		return nil, err
	}
	rng, err := spanToRange(pkg, spn)
	if err != nil {
		return nil, err
	}
	diag := &source.Diagnostic{
		URI:      spn.URI(),
		Range:    rng,
		Severity: protocol.SeverityError,
		Source:   source.TypeError,
		Message:  e.primary.Msg,
	}
	if code != 0 {
		diag.Code = code.String()
		diag.CodeHref = typesCodeHref(snapshot, code)
	}
	switch code {
	case typesinternal.UnusedVar, typesinternal.UnusedImport:
		diag.Tags = append(diag.Tags, protocol.Unnecessary)
	}

	for _, secondary := range e.secondaries {
		_, secondarySpan, err := typeErrorData(snapshot.FileSet(), pkg, secondary)
		if err != nil {
			return nil, err
		}
		rng, err := spanToRange(pkg, secondarySpan)
		if err != nil {
			return nil, err
		}
		diag.Related = append(diag.Related, source.RelatedInformation{
			URI:     secondarySpan.URI(),
			Range:   rng,
			Message: secondary.Msg,
		})
	}

	if match := importErrorRe.FindStringSubmatch(e.primary.Msg); match != nil {
		diag.SuggestedFixes, err = goGetQuickFixes(snapshot, spn.URI(), match[1])
		if err != nil {
			return nil, err
		}
	}
	if match := unsupportedFeatureRe.FindStringSubmatch(e.primary.Msg); match != nil {
		diag.SuggestedFixes, err = editGoDirectiveQuickFix(snapshot, spn.URI(), match[1])
		if err != nil {
			return nil, err
		}
	}
	return []*source.Diagnostic{diag}, nil
}

func goGetQuickFixes(snapshot *snapshot, uri span.URI, pkg string) ([]source.SuggestedFix, error) {
	// Go get only supports module mode for now.
	if snapshot.workspaceMode()&moduleMode == 0 {
		return nil, nil
	}
	title := fmt.Sprintf("go get package %v", pkg)
	cmd, err := command.NewGoGetPackageCommand(title, command.GoGetPackageArgs{
		URI:        protocol.URIFromSpanURI(uri),
		AddRequire: true,
		Pkg:        pkg,
	})
	if err != nil {
		return nil, err
	}
	return []source.SuggestedFix{source.SuggestedFixFromCommand(cmd, protocol.QuickFix)}, nil
}

func editGoDirectiveQuickFix(snapshot *snapshot, uri span.URI, version string) ([]source.SuggestedFix, error) {
	// Go mod edit only supports module mode.
	if snapshot.workspaceMode()&moduleMode == 0 {
		return nil, nil
	}
	title := fmt.Sprintf("go mod edit -go=%s", version)
	cmd, err := command.NewEditGoDirectiveCommand(title, command.EditGoDirectiveArgs{
		URI:     protocol.URIFromSpanURI(uri),
		Version: version,
	})
	if err != nil {
		return nil, err
	}
	return []source.SuggestedFix{source.SuggestedFixFromCommand(cmd, protocol.QuickFix)}, nil
}

func analysisDiagnosticDiagnostics(snapshot *snapshot, pkg *pkg, a *analysis.Analyzer, e *analysis.Diagnostic) ([]*source.Diagnostic, error) {
	var srcAnalyzer *source.Analyzer
	// Find the analyzer that generated this diagnostic.
	for _, sa := range source.EnabledAnalyzers(snapshot) {
		if a == sa.Analyzer {
			srcAnalyzer = sa
			break
		}
	}
	tokFile := snapshot.FileSet().File(e.Pos)
	if tokFile == nil {
		return nil, bug.Errorf("no file for position of %q diagnostic", e.Category)
	}
	end := e.End
	if !end.IsValid() {
		end = e.Pos
	}
	spn, err := span.NewRange(tokFile, e.Pos, end).Span()
	if err != nil {
		return nil, err
	}
	rng, err := spanToRange(pkg, spn)
	if err != nil {
		return nil, err
	}
	kinds := srcAnalyzer.ActionKind
	if len(srcAnalyzer.ActionKind) == 0 {
		kinds = append(kinds, protocol.QuickFix)
	}
	fixes, err := suggestedAnalysisFixes(snapshot, pkg, e, kinds)
	if err != nil {
		return nil, err
	}
	if srcAnalyzer.Fix != "" {
		cmd, err := command.NewApplyFixCommand(e.Message, command.ApplyFixArgs{
			URI:   protocol.URIFromSpanURI(spn.URI()),
			Range: rng,
			Fix:   srcAnalyzer.Fix,
		})
		if err != nil {
			return nil, err
		}
		for _, kind := range kinds {
			fixes = append(fixes, source.SuggestedFixFromCommand(cmd, kind))
		}
	}
	related, err := relatedInformation(pkg, snapshot.FileSet(), e)
	if err != nil {
		return nil, err
	}

	severity := srcAnalyzer.Severity
	if severity == 0 {
		severity = protocol.SeverityWarning
	}
	diag := &source.Diagnostic{
		URI:            spn.URI(),
		Range:          rng,
		Severity:       severity,
		Source:         source.AnalyzerErrorKind(e.Category),
		Message:        e.Message,
		Related:        related,
		SuggestedFixes: fixes,
		Analyzer:       srcAnalyzer,
	}
	// If the fixes only delete code, assume that the diagnostic is reporting dead code.
	if onlyDeletions(fixes) {
		diag.Tags = []protocol.DiagnosticTag{protocol.Unnecessary}
	}
	return []*source.Diagnostic{diag}, nil
}

// onlyDeletions returns true if all of the suggested fixes are deletions.
func onlyDeletions(fixes []source.SuggestedFix) bool {
	for _, fix := range fixes {
		if fix.Command != nil {
			return false
		}
		for _, edits := range fix.Edits {
			for _, edit := range edits {
				if edit.NewText != "" {
					return false
				}
				if protocol.ComparePosition(edit.Range.Start, edit.Range.End) == 0 {
					return false
				}
			}
		}
	}
	return len(fixes) > 0
}

func typesCodeHref(snapshot *snapshot, code typesinternal.ErrorCode) string {
	target := snapshot.View().Options().LinkTarget
	return source.BuildLink(target, "golang.org/x/tools/internal/typesinternal", code.String())
}

func suggestedAnalysisFixes(snapshot *snapshot, pkg *pkg, diag *analysis.Diagnostic, kinds []protocol.CodeActionKind) ([]source.SuggestedFix, error) {
	var fixes []source.SuggestedFix
	for _, fix := range diag.SuggestedFixes {
		edits := make(map[span.URI][]protocol.TextEdit)
		for _, e := range fix.TextEdits {
			tokFile := snapshot.FileSet().File(e.Pos)
			if tokFile == nil {
				return nil, bug.Errorf("no file for edit position")
			}
			end := e.End
			if !end.IsValid() {
				end = e.Pos
			}
			spn, err := span.NewRange(tokFile, e.Pos, end).Span()
			if err != nil {
				return nil, err
			}
			rng, err := spanToRange(pkg, spn)
			if err != nil {
				return nil, err
			}
			edits[spn.URI()] = append(edits[spn.URI()], protocol.TextEdit{
				Range:   rng,
				NewText: string(e.NewText),
			})
		}
		for _, kind := range kinds {
			fixes = append(fixes, source.SuggestedFix{
				Title:      fix.Message,
				Edits:      edits,
				ActionKind: kind,
			})
		}

	}
	return fixes, nil
}

func relatedInformation(pkg *pkg, fset *token.FileSet, diag *analysis.Diagnostic) ([]source.RelatedInformation, error) {
	var out []source.RelatedInformation
	for _, related := range diag.Related {
		tokFile := fset.File(related.Pos)
		if tokFile == nil {
			return nil, bug.Errorf("no file for %q diagnostic position", diag.Category)
		}
		end := related.End
		if !end.IsValid() {
			end = related.Pos
		}
		spn, err := span.NewRange(tokFile, related.Pos, end).Span()
		if err != nil {
			return nil, err
		}
		rng, err := spanToRange(pkg, spn)
		if err != nil {
			return nil, err
		}
		out = append(out, source.RelatedInformation{
			URI:     spn.URI(),
			Range:   rng,
			Message: related.Message,
		})
	}
	return out, nil
}

func typeErrorData(fset *token.FileSet, pkg *pkg, terr types.Error) (typesinternal.ErrorCode, span.Span, error) {
	ecode, start, end, ok := typesinternal.ReadGo116ErrorData(terr)
	if !ok {
		start, end = terr.Pos, terr.Pos
		ecode = 0
	}
	// go/types may return invalid positions in some cases, such as
	// in errors on tokens missing from the syntax tree.
	if !start.IsValid() {
		return 0, span.Span{}, fmt.Errorf("type error (%q, code %d, go116=%t) without position", terr.Msg, ecode, ok)
	}
	// go/types errors retain their FileSet.
	// Sanity-check that we're using the right one.
	if fset != terr.Fset {
		return 0, span.Span{}, bug.Errorf("wrong FileSet for type error")
	}
	posn := fset.Position(start)
	if !posn.IsValid() {
		return 0, span.Span{}, fmt.Errorf("position %d of type error %q (code %q) not found in FileSet", start, start, terr)
	}
	pgf, err := pkg.File(span.URIFromPath(posn.Filename))
	if err != nil {
		return 0, span.Span{}, err
	}
	if !end.IsValid() || end == start {
		end = analysisinternal.TypeErrorEndPos(fset, pgf.Src, start)
	}
	spn, err := parsedGoSpan(pgf, start, end)
	if err != nil {
		return 0, span.Span{}, err
	}
	return ecode, spn, nil
}

func parsedGoSpan(pgf *source.ParsedGoFile, start, end token.Pos) (span.Span, error) {
	return span.FileSpan(pgf.Mapper.TokFile, pgf.Mapper.TokFile, start, end)
}

// spanToRange converts a span.Span to a protocol.Range,
// assuming that the span belongs to the package whose diagnostics are being computed.
func spanToRange(pkg *pkg, spn span.Span) (protocol.Range, error) {
	pgf, err := pkg.File(spn.URI())
	if err != nil {
		return protocol.Range{}, err
	}
	return pgf.Mapper.Range(spn)
}

// parseGoListError attempts to parse a standard `go list` error message
// by stripping off the trailing error message.
//
// It works only on errors whose message is prefixed by colon,
// followed by a space (": "). For example:
//
//	attributes.go:13:1: expected 'package', found 'type'
func parseGoListError(input, wd string) span.Span {
	input = strings.TrimSpace(input)
	msgIndex := strings.Index(input, ": ")
	if msgIndex < 0 {
		return span.Parse(input)
	}
	return span.ParseInDir(input[:msgIndex], wd)
}

func parseGoListImportCycleError(snapshot *snapshot, e packages.Error, pkg *pkg) (string, span.Span, bool) {
	re := regexp.MustCompile(`(.*): import stack: \[(.+)\]`)
	matches := re.FindStringSubmatch(strings.TrimSpace(e.Msg))
	if len(matches) < 3 {
		return e.Msg, span.Span{}, false
	}
	msg := matches[1]
	importList := strings.Split(matches[2], " ")
	// Since the error is relative to the current package. The import that is causing
	// the import cycle error is the second one in the list.
	if len(importList) < 2 {
		return msg, span.Span{}, false
	}
	// Imports have quotation marks around them.
	circImp := strconv.Quote(importList[1])
	for _, cgf := range pkg.compiledGoFiles {
		// Search file imports for the import that is causing the import cycle.
		for _, imp := range cgf.File.Imports {
			if imp.Path.Value == circImp {
				spn, err := span.NewRange(cgf.Tok, imp.Pos(), imp.End()).Span()
				if err != nil {
					return msg, span.Span{}, false
				}
				return msg, spn, true
			}
		}
	}
	return msg, span.Span{}, false
}
