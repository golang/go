// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

type SuggestedFix struct {
	Title   string
	Edits   map[span.URI][]protocol.TextEdit
	Command *protocol.Command
}

type RelatedInformation struct {
	URI     span.URI
	Range   protocol.Range
	Message string
}

func GetTypeCheckDiagnostics(ctx context.Context, snapshot Snapshot, pkg Package) TypeCheckDiagnostics {
	onlyIgnoredFiles := true
	for _, pgf := range pkg.CompiledGoFiles() {
		onlyIgnoredFiles = onlyIgnoredFiles && snapshot.IgnoredFile(pgf.URI)
	}
	if onlyIgnoredFiles {
		return TypeCheckDiagnostics{}
	}
	return typeCheckDiagnostics(ctx, snapshot, pkg)
}

func Analyze(ctx context.Context, snapshot Snapshot, pkg Package, typeCheckResult TypeCheckDiagnostics) (map[span.URI][]*Diagnostic, error) {
	// Exit early if the context has been canceled. This also protects us
	// from a race on Options, see golang/go#36699.
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	// If we don't have any list or parse errors, run analyses.
	analyzers := pickAnalyzers(snapshot, typeCheckResult.HasTypeErrors)
	analysisDiagnostics, err := snapshot.Analyze(ctx, pkg.ID(), analyzers...)
	if err != nil {
		return nil, err
	}
	analysisDiagnostics = cloneDiagnostics(analysisDiagnostics)

	reports := emptyDiagnostics(pkg)
	// Report diagnostics and errors from root analyzers.
	for _, diag := range analysisDiagnostics {
		// If the diagnostic comes from a "convenience" analyzer, it is not
		// meant to provide diagnostics, but rather only suggested fixes.
		// Skip these types of errors in diagnostics; we will use their
		// suggested fixes when providing code actions.
		if isConvenienceAnalyzer(string(diag.Source)) {
			continue
		}
		// This is a bit of a hack, but clients > 3.15 will be able to grey out unnecessary code.
		// If we are deleting code as part of all of our suggested fixes, assume that this is dead code.
		// TODO(golang/go#34508): Return these codes from the diagnostics themselves.
		var tags []protocol.DiagnosticTag
		if onlyDeletions(diag.SuggestedFixes) {
			tags = append(tags, protocol.Unnecessary)
		}
		// Type error analyzers only alter the tags for existing type errors.
		if _, ok := snapshot.View().Options().TypeErrorAnalyzers[string(diag.Source)]; ok {
			existingDiagnostics := typeCheckResult.Diagnostics[diag.URI]
			for _, existing := range existingDiagnostics {
				if r := protocol.CompareRange(diag.Range, existing.Range); r != 0 {
					continue
				}
				if diag.Message != existing.Message {
					continue
				}
				existing.Tags = append(existing.Tags, tags...)
			}
		} else {
			diag.Tags = append(diag.Tags, tags...)
			reports[diag.URI] = append(reports[diag.URI], diag)
		}
	}
	return reports, nil
}

// cloneDiagnostics makes a shallow copy of diagnostics so that Analyze
// can add tags to them without affecting the cached diagnostics.
func cloneDiagnostics(diags []*Diagnostic) []*Diagnostic {
	result := []*Diagnostic{}
	for _, d := range diags {
		clone := *d
		result = append(result, &clone)
	}
	return result
}

func pickAnalyzers(snapshot Snapshot, hadTypeErrors bool) []*analysis.Analyzer {
	// Always run convenience analyzers.
	categories := []map[string]Analyzer{snapshot.View().Options().ConvenienceAnalyzers}
	// If we had type errors, only run type error analyzers.
	if hadTypeErrors {
		categories = append(categories, snapshot.View().Options().TypeErrorAnalyzers)
	} else {
		categories = append(categories, snapshot.View().Options().DefaultAnalyzers, snapshot.View().Options().StaticcheckAnalyzers)
	}
	var analyzers []*analysis.Analyzer
	for _, m := range categories {
		for _, a := range m {
			if a.IsEnabled(snapshot.View()) {
				analyzers = append(analyzers, a.Analyzer)
			}
		}
	}
	return analyzers
}

func FileDiagnostics(ctx context.Context, snapshot Snapshot, uri span.URI) (VersionedFileIdentity, []*Diagnostic, error) {
	fh, err := snapshot.GetVersionedFile(ctx, uri)
	if err != nil {
		return VersionedFileIdentity{}, nil, err
	}
	pkg, _, err := GetParsedFile(ctx, snapshot, fh, NarrowestPackage)
	if err != nil {
		return VersionedFileIdentity{}, nil, err
	}
	typeCheckResults := GetTypeCheckDiagnostics(ctx, snapshot, pkg)
	diagnostics := typeCheckResults.Diagnostics[fh.URI()]
	if !typeCheckResults.HasParseOrListErrors {
		reports, err := Analyze(ctx, snapshot, pkg, typeCheckResults)
		if err != nil {
			return VersionedFileIdentity{}, nil, err
		}
		diagnostics = append(diagnostics, reports[fh.URI()]...)
	}
	return fh.VersionedFileIdentity(), diagnostics, nil
}

type TypeCheckDiagnostics struct {
	HasTypeErrors        bool
	HasParseOrListErrors bool
	Diagnostics          map[span.URI][]*Diagnostic
}

type diagnosticSet struct {
	listErrors, parseErrors, typeErrors []*Diagnostic
}

func typeCheckDiagnostics(ctx context.Context, snapshot Snapshot, pkg Package) TypeCheckDiagnostics {
	ctx, done := event.Start(ctx, "source.typeCheckDiagnostics", tag.Package.Of(pkg.ID()))
	_ = ctx // circumvent SA4006
	defer done()

	diagSets := make(map[span.URI]*diagnosticSet)
	for _, diag := range pkg.GetDiagnostics() {
		set, ok := diagSets[diag.URI]
		if !ok {
			set = &diagnosticSet{}
			diagSets[diag.URI] = set
		}
		switch diag.Source {
		case ParseError:
			set.parseErrors = append(set.parseErrors, diag)
		case TypeError:
			set.typeErrors = append(set.typeErrors, diag)
		case ListError:
			set.listErrors = append(set.listErrors, diag)
		}
	}
	typecheck := TypeCheckDiagnostics{
		Diagnostics: emptyDiagnostics(pkg),
	}
	for uri, set := range diagSets {
		// Don't report type errors if there are parse errors or list errors.
		diags := set.typeErrors
		switch {
		case len(set.parseErrors) > 0:
			typecheck.HasParseOrListErrors = true
			diags = set.parseErrors
		case len(set.listErrors) > 0:
			typecheck.HasParseOrListErrors = true
			if len(pkg.MissingDependencies()) > 0 {
				diags = set.listErrors
			}
		case len(set.typeErrors) > 0:
			typecheck.HasTypeErrors = true
		}
		typecheck.Diagnostics[uri] = cloneDiagnostics(diags)
	}
	return typecheck
}

func emptyDiagnostics(pkg Package) map[span.URI][]*Diagnostic {
	diags := map[span.URI][]*Diagnostic{}
	for _, pgf := range pkg.CompiledGoFiles() {
		if _, ok := diags[pgf.URI]; !ok {
			diags[pgf.URI] = nil
		}
	}
	return diags
}

// onlyDeletions returns true if all of the suggested fixes are deletions.
func onlyDeletions(fixes []SuggestedFix) bool {
	for _, fix := range fixes {
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

func isConvenienceAnalyzer(category string) bool {
	for _, a := range DefaultOptions().ConvenienceAnalyzers {
		if category == a.Analyzer.Name {
			return true
		}
	}
	return false
}
