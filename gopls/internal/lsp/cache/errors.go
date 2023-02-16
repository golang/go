// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

// This file defines routines to convert diagnostics from go list, go
// get, go/packages, parsing, type checking, and analysis into
// source.Diagnostic form, and suggesting quick fixes.

import (
	"context"
	"fmt"
	"go/scanner"
	"go/token"
	"go/types"
	"log"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/typesinternal"
)

// goPackagesErrorDiagnostics translates the given go/packages Error into a
// diagnostic, using the provided metadata and filesource.
//
// The slice of diagnostics may be empty.
func goPackagesErrorDiagnostics(ctx context.Context, e packages.Error, m *source.Metadata, fs source.FileSource) ([]*source.Diagnostic, error) {
	if diag, err := parseGoListImportCycleError(ctx, e, m, fs); err != nil {
		return nil, err
	} else if diag != nil {
		return []*source.Diagnostic{diag}, nil
	}

	var spn span.Span
	if e.Pos == "" {
		spn = parseGoListError(e.Msg, m.LoadDir)
		// We may not have been able to parse a valid span. Apply the errors to all files.
		if _, err := spanToRange(ctx, fs, spn); err != nil {
			var diags []*source.Diagnostic
			for _, uri := range m.CompiledGoFiles {
				diags = append(diags, &source.Diagnostic{
					URI:      uri,
					Severity: protocol.SeverityError,
					Source:   source.ListError,
					Message:  e.Msg,
				})
			}
			return diags, nil
		}
	} else {
		spn = span.ParseInDir(e.Pos, m.LoadDir)
	}

	// TODO(rfindley): in some cases the go command outputs invalid spans, for
	// example (from TestGoListErrors):
	//
	//   package a
	//   import
	//
	// In this case, the go command will complain about a.go:2:8, which is after
	// the trailing newline but still considered to be on the second line, most
	// likely because *token.File lacks information about newline termination.
	//
	// We could do better here by handling that case.
	rng, err := spanToRange(ctx, fs, spn)
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

func parseErrorDiagnostics(snapshot *snapshot, pkg *syntaxPackage, errList scanner.ErrorList) ([]*source.Diagnostic, error) {
	// The first parser error is likely the root cause of the problem.
	if errList.Len() <= 0 {
		return nil, fmt.Errorf("no errors in %v", errList)
	}
	e := errList[0]
	pgf, err := pkg.File(span.URIFromPath(e.Pos.Filename))
	if err != nil {
		return nil, err
	}
	rng, err := pgf.Mapper.OffsetRange(e.Pos.Offset, e.Pos.Offset)
	if err != nil {
		return nil, err
	}
	return []*source.Diagnostic{{
		URI:      pgf.URI,
		Range:    rng,
		Severity: protocol.SeverityError,
		Source:   source.ParseError,
		Message:  e.Msg,
	}}, nil
}

var importErrorRe = regexp.MustCompile(`could not import ([^\s]+)`)
var unsupportedFeatureRe = regexp.MustCompile(`.*require.* go(\d+\.\d+) or later`)

func typeErrorDiagnostics(snapshot *snapshot, pkg *syntaxPackage, e extendedError) ([]*source.Diagnostic, error) {
	code, loc, err := typeErrorData(pkg, e.primary)
	if err != nil {
		return nil, err
	}
	diag := &source.Diagnostic{
		URI:      loc.URI.SpanURI(),
		Range:    loc.Range,
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
		_, secondaryLoc, err := typeErrorData(pkg, secondary)
		if err != nil {
			return nil, err
		}
		diag.Related = append(diag.Related, protocol.DiagnosticRelatedInformation{
			Location: secondaryLoc,
			Message:  secondary.Msg,
		})
	}

	if match := importErrorRe.FindStringSubmatch(e.primary.Msg); match != nil {
		diag.SuggestedFixes, err = goGetQuickFixes(snapshot.moduleMode(), loc.URI.SpanURI(), match[1])
		if err != nil {
			return nil, err
		}
	}
	if match := unsupportedFeatureRe.FindStringSubmatch(e.primary.Msg); match != nil {
		diag.SuggestedFixes, err = editGoDirectiveQuickFix(snapshot, loc.URI.SpanURI(), match[1])
		if err != nil {
			return nil, err
		}
	}
	return []*source.Diagnostic{diag}, nil
}

func goGetQuickFixes(moduleMode bool, uri span.URI, pkg string) ([]source.SuggestedFix, error) {
	if !moduleMode {
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

// toSourceDiagnostic converts a gobDiagnostic to "source" form.
func toSourceDiagnostic(srcAnalyzer *source.Analyzer, gobDiag *gobDiagnostic) *source.Diagnostic {
	var related []protocol.DiagnosticRelatedInformation
	for _, gobRelated := range gobDiag.Related {
		related = append(related, protocol.DiagnosticRelatedInformation(gobRelated))
	}

	kinds := srcAnalyzer.ActionKind
	if len(srcAnalyzer.ActionKind) == 0 {
		kinds = append(kinds, protocol.QuickFix)
	}
	fixes := suggestedAnalysisFixes(gobDiag, kinds)
	if srcAnalyzer.Fix != "" {
		cmd, err := command.NewApplyFixCommand(gobDiag.Message, command.ApplyFixArgs{
			URI:   gobDiag.Location.URI,
			Range: gobDiag.Location.Range,
			Fix:   srcAnalyzer.Fix,
		})
		if err != nil {
			// JSON marshalling of these argument values cannot fail.
			log.Fatalf("internal error in NewApplyFixCommand: %v", err)
		}
		for _, kind := range kinds {
			fixes = append(fixes, source.SuggestedFixFromCommand(cmd, kind))
		}
	}

	severity := srcAnalyzer.Severity
	if severity == 0 {
		severity = protocol.SeverityWarning
	}

	diag := &source.Diagnostic{
		// TODO(adonovan): is this sound? See dual conversion in posToLocation.
		URI:            span.URI(gobDiag.Location.URI),
		Range:          gobDiag.Location.Range,
		Severity:       severity,
		Source:         source.AnalyzerErrorKind(gobDiag.Category),
		Message:        gobDiag.Message,
		Related:        related,
		SuggestedFixes: fixes,
	}
	// If the fixes only delete code, assume that the diagnostic is reporting dead code.
	if onlyDeletions(fixes) {
		diag.Tags = []protocol.DiagnosticTag{protocol.Unnecessary}
	}
	return diag
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

func suggestedAnalysisFixes(diag *gobDiagnostic, kinds []protocol.CodeActionKind) []source.SuggestedFix {
	var fixes []source.SuggestedFix
	for _, fix := range diag.SuggestedFixes {
		edits := make(map[span.URI][]protocol.TextEdit)
		for _, e := range fix.TextEdits {
			uri := span.URI(e.Location.URI)
			edits[uri] = append(edits[uri], protocol.TextEdit{
				Range:   e.Location.Range,
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
	return fixes
}

func typeErrorData(pkg *syntaxPackage, terr types.Error) (typesinternal.ErrorCode, protocol.Location, error) {
	ecode, start, end, ok := typesinternal.ReadGo116ErrorData(terr)
	if !ok {
		start, end = terr.Pos, terr.Pos
		ecode = 0
	}
	// go/types may return invalid positions in some cases, such as
	// in errors on tokens missing from the syntax tree.
	if !start.IsValid() {
		return 0, protocol.Location{}, fmt.Errorf("type error (%q, code %d, go116=%t) without position", terr.Msg, ecode, ok)
	}
	// go/types errors retain their FileSet.
	// Sanity-check that we're using the right one.
	fset := pkg.fset
	if fset != terr.Fset {
		return 0, protocol.Location{}, bug.Errorf("wrong FileSet for type error")
	}
	posn := safetoken.StartPosition(fset, start)
	if !posn.IsValid() {
		return 0, protocol.Location{}, fmt.Errorf("position %d of type error %q (code %q) not found in FileSet", start, start, terr)
	}
	pgf, err := pkg.File(span.URIFromPath(posn.Filename))
	if err != nil {
		return 0, protocol.Location{}, err
	}
	if !end.IsValid() || end == start {
		end = analysisinternal.TypeErrorEndPos(fset, pgf.Src, start)
	}
	loc, err := pgf.Mapper.PosLocation(pgf.Tok, start, end)
	return ecode, loc, err
}

// spanToRange converts a span.Span to a protocol.Range, by mapping content
// contained in the provided FileSource.
func spanToRange(ctx context.Context, fs source.FileSource, spn span.Span) (protocol.Range, error) {
	uri := spn.URI()
	fh, err := fs.GetFile(ctx, uri)
	if err != nil {
		return protocol.Range{}, err
	}
	content, err := fh.Read()
	if err != nil {
		return protocol.Range{}, err
	}
	mapper := protocol.NewMapper(uri, content)
	return mapper.SpanRange(spn)
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

// parseGoListImportCycleError attempts to parse the given go/packages error as
// an import cycle, returning a diagnostic if successful.
//
// If the error is not detected as an import cycle error, it returns nil, nil.
func parseGoListImportCycleError(ctx context.Context, e packages.Error, m *source.Metadata, fs source.FileSource) (*source.Diagnostic, error) {
	re := regexp.MustCompile(`(.*): import stack: \[(.+)\]`)
	matches := re.FindStringSubmatch(strings.TrimSpace(e.Msg))
	if len(matches) < 3 {
		return nil, nil
	}
	msg := matches[1]
	importList := strings.Split(matches[2], " ")
	// Since the error is relative to the current package. The import that is causing
	// the import cycle error is the second one in the list.
	if len(importList) < 2 {
		return nil, nil
	}
	// Imports have quotation marks around them.
	circImp := strconv.Quote(importList[1])
	for _, uri := range m.CompiledGoFiles {
		pgf, err := parseGoURI(ctx, fs, uri, source.ParseHeader)
		if err != nil {
			return nil, err
		}
		// Search file imports for the import that is causing the import cycle.
		for _, imp := range pgf.File.Imports {
			if imp.Path.Value == circImp {
				rng, err := pgf.NodeMappedRange(imp)
				if err != nil {
					return nil, nil
				}

				return &source.Diagnostic{
					URI:      pgf.URI,
					Range:    rng.Range(),
					Severity: protocol.SeverityError,
					Source:   source.ListError,
					Message:  msg,
				}, nil
			}
		}
	}
	return nil, nil
}

// parseGoURI is a helper to parse the Go file at the given URI from the file
// source fs. The resulting syntax and token.File belong to an ephemeral,
// encapsulated FileSet, so this file stands only on its own: it's not suitable
// to use in a list of file of a package, for example.
//
// It returns an error if the file could not be read.
func parseGoURI(ctx context.Context, fs source.FileSource, uri span.URI, mode source.ParseMode) (*source.ParsedGoFile, error) {
	fh, err := fs.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	return parseGoImpl(ctx, token.NewFileSet(), fh, source.ParseHeader)
}

// parseModURI is a helper to parse the Mod file at the given URI from the file
// source fs.
//
// It returns an error if the file could not be read.
func parseModURI(ctx context.Context, fs source.FileSource, uri span.URI) (*source.ParsedModule, error) {
	fh, err := fs.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	return parseModImpl(ctx, fh)
}
