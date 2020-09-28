// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

type Diagnostic struct {
	Range    protocol.Range
	Message  string
	Source   string
	Severity protocol.DiagnosticSeverity
	Tags     []protocol.DiagnosticTag

	Related []RelatedInformation
}

type SuggestedFix struct {
	Title string
	Edits map[span.URI][]protocol.TextEdit
}

type RelatedInformation struct {
	URI     span.URI
	Range   protocol.Range
	Message string
}

func Diagnostics(ctx context.Context, snapshot Snapshot, pkg Package, withAnalysis bool) (map[VersionedFileIdentity][]*Diagnostic, bool, error) {
	onlyIgnoredFiles := true
	for _, pgf := range pkg.CompiledGoFiles() {
		onlyIgnoredFiles = onlyIgnoredFiles && snapshot.IgnoredFile(pgf.URI)
	}
	if onlyIgnoredFiles {
		return nil, false, nil
	}

	// If we are missing dependencies, it may because the user's workspace is
	// not correctly configured. Report errors, if possible.
	var warn bool
	if len(pkg.MissingDependencies()) > 0 {
		warn = true
	}
	// If we have a package with a single file and errors about "undeclared" symbols,
	// we may have an ad-hoc package with multiple files. Show a warning message.
	// TODO(golang/go#36416): Remove this when golang.org/cl/202277 is merged.
	if len(pkg.CompiledGoFiles()) == 1 && hasUndeclaredErrors(pkg) {
		warn = true
	}
	// Prepare the reports we will send for the files in this package.
	reports := make(map[VersionedFileIdentity][]*Diagnostic)
	for _, pgf := range pkg.CompiledGoFiles() {
		clearReports(ctx, snapshot, reports, pgf.URI)
	}
	// Prepare any additional reports for the errors in this package.
	for _, e := range pkg.GetErrors() {
		// We only need to handle lower-level errors.
		if e.Kind != ListError {
			continue
		}
		// If no file is associated with the error, pick an open file from the package.
		if e.URI.Filename() == "" {
			for _, pgf := range pkg.CompiledGoFiles() {
				if snapshot.IsOpen(pgf.URI) {
					e.URI = pgf.URI
				}
			}
		}
		clearReports(ctx, snapshot, reports, e.URI)
	}
	// Run diagnostics for the package that this URI belongs to.
	hadDiagnostics, hadTypeErrors, err := diagnostics(ctx, snapshot, reports, pkg, len(pkg.MissingDependencies()) > 0)
	if err != nil {
		return nil, warn, err
	}
	if hadDiagnostics || !withAnalysis {
		return reports, warn, nil
	}
	// Exit early if the context has been canceled. This also protects us
	// from a race on Options, see golang/go#36699.
	if ctx.Err() != nil {
		return nil, warn, ctx.Err()
	}
	// If we don't have any list or parse errors, run analyses.
	analyzers := pickAnalyzers(snapshot, hadTypeErrors)
	if err := analyses(ctx, snapshot, reports, pkg, analyzers); err != nil {
		event.Error(ctx, "analyses failed", err, tag.Snapshot.Of(snapshot.ID()), tag.Package.Of(pkg.ID()))
		if ctx.Err() != nil {
			return nil, warn, ctx.Err()
		}
	}
	return reports, warn, nil
}

func pickAnalyzers(snapshot Snapshot, hadTypeErrors bool) map[string]Analyzer {
	analyzers := make(map[string]Analyzer)

	// Always run convenience analyzers.
	for k, v := range snapshot.View().Options().ConvenienceAnalyzers {
		analyzers[k] = v
	}
	// If we had type errors, only run type error analyzers.
	if hadTypeErrors {
		for k, v := range snapshot.View().Options().TypeErrorAnalyzers {
			analyzers[k] = v
		}
		return analyzers
	}
	for k, v := range snapshot.View().Options().DefaultAnalyzers {
		analyzers[k] = v
	}
	for k, v := range snapshot.View().Options().StaticcheckAnalyzers {
		analyzers[k] = v
	}
	return analyzers
}

func FileDiagnostics(ctx context.Context, snapshot Snapshot, uri span.URI) (VersionedFileIdentity, []*Diagnostic, error) {
	fh, err := snapshot.GetFile(ctx, uri)
	if err != nil {
		return VersionedFileIdentity{}, nil, err
	}
	pkg, _, err := GetParsedFile(ctx, snapshot, fh, NarrowestPackage)
	if err != nil {
		return VersionedFileIdentity{}, nil, err
	}
	reports, _, err := Diagnostics(ctx, snapshot, pkg, true)
	if err != nil {
		return VersionedFileIdentity{}, nil, err
	}
	diagnostics, ok := reports[fh.VersionedFileIdentity()]
	if !ok {
		return VersionedFileIdentity{}, nil, errors.Errorf("no diagnostics for %s", uri)
	}
	return fh.VersionedFileIdentity(), diagnostics, nil
}

type diagnosticSet struct {
	listErrors, parseErrors, typeErrors []*Diagnostic
}

func diagnostics(ctx context.Context, snapshot Snapshot, reports map[VersionedFileIdentity][]*Diagnostic, pkg Package, hasMissingDeps bool) (bool, bool, error) {
	ctx, done := event.Start(ctx, "source.diagnostics", tag.Package.Of(pkg.ID()))
	_ = ctx // circumvent SA4006
	defer done()

	diagSets := make(map[span.URI]*diagnosticSet)
	for _, e := range pkg.GetErrors() {
		diag := &Diagnostic{
			Message:  e.Message,
			Range:    e.Range,
			Severity: protocol.SeverityError,
			Related:  e.Related,
		}
		set, ok := diagSets[e.URI]
		if !ok {
			set = &diagnosticSet{}
			diagSets[e.URI] = set
		}
		switch e.Kind {
		case ParseError:
			set.parseErrors = append(set.parseErrors, diag)
			diag.Source = "syntax"
		case TypeError:
			set.typeErrors = append(set.typeErrors, diag)
			diag.Source = "compiler"
		case ListError:
			set.listErrors = append(set.listErrors, diag)
			diag.Source = "go list"
		}
	}
	var nonEmptyDiagnostics, hasTypeErrors bool // track if we actually send non-empty diagnostics
	for uri, set := range diagSets {
		// Don't report type errors if there are parse errors or list errors.
		diags := set.typeErrors
		if len(set.parseErrors) > 0 {
			diags, nonEmptyDiagnostics = set.parseErrors, true
		} else if len(set.listErrors) > 0 {
			// Only show list errors if the package has missing dependencies.
			if hasMissingDeps {
				diags, nonEmptyDiagnostics = set.listErrors, true
			}
		} else if len(set.typeErrors) > 0 {
			hasTypeErrors = true
		}
		if err := addReports(ctx, snapshot, reports, uri, diags...); err != nil {
			return false, false, err
		}
	}
	return nonEmptyDiagnostics, hasTypeErrors, nil
}

func analyses(ctx context.Context, snapshot Snapshot, reports map[VersionedFileIdentity][]*Diagnostic, pkg Package, analyses map[string]Analyzer) error {
	var analyzers []*analysis.Analyzer
	for _, a := range analyses {
		if !a.IsEnabled(snapshot.View()) {
			continue
		}
		analyzers = append(analyzers, a.Analyzer)
	}
	analysisErrors, err := snapshot.Analyze(ctx, pkg.ID(), analyzers...)
	if err != nil {
		return err
	}

	// Report diagnostics and errors from root analyzers.
	for _, e := range analysisErrors {
		// If the diagnostic comes from a "convenience" analyzer, it is not
		// meant to provide diagnostics, but rather only suggested fixes.
		// Skip these types of errors in diagnostics; we will use their
		// suggested fixes when providing code actions.
		if isConvenienceAnalyzer(snapshot.View().Options(), e.Category) {
			continue
		}
		// This is a bit of a hack, but clients > 3.15 will be able to grey out unnecessary code.
		// If we are deleting code as part of all of our suggested fixes, assume that this is dead code.
		// TODO(golang/go#34508): Return these codes from the diagnostics themselves.
		var tags []protocol.DiagnosticTag
		if onlyDeletions(e.SuggestedFixes) {
			tags = append(tags, protocol.Unnecessary)
		}
		if err := addReports(ctx, snapshot, reports, e.URI, &Diagnostic{
			Range:    e.Range,
			Message:  e.Message,
			Source:   e.Category,
			Severity: protocol.SeverityWarning,
			Tags:     tags,
			Related:  e.Related,
		}); err != nil {
			return err
		}
	}
	return nil
}

func clearReports(ctx context.Context, snapshot Snapshot, reports map[VersionedFileIdentity][]*Diagnostic, uri span.URI) {
	fh := snapshot.FindFile(uri)
	if fh == nil {
		return
	}
	reports[fh.VersionedFileIdentity()] = []*Diagnostic{}
}

func addReports(ctx context.Context, snapshot Snapshot, reports map[VersionedFileIdentity][]*Diagnostic, uri span.URI, diagnostics ...*Diagnostic) error {
	fh := snapshot.FindFile(uri)
	if fh == nil {
		return nil
	}
	existingDiagnostics, ok := reports[fh.VersionedFileIdentity()]
	if !ok {
		return fmt.Errorf("diagnostics for unexpected file %s", uri)
	}
	if len(diagnostics) == 1 {
		d1 := diagnostics[0]
		if _, ok := snapshot.View().Options().TypeErrorAnalyzers[d1.Source]; ok {
			for i, d2 := range existingDiagnostics {
				if r := protocol.CompareRange(d1.Range, d2.Range); r != 0 {
					continue
				}
				if d1.Message != d2.Message {
					continue
				}
				reports[fh.VersionedFileIdentity()][i].Tags = append(reports[fh.VersionedFileIdentity()][i].Tags, d1.Tags...)
			}
			return nil
		}
	}
	reports[fh.VersionedFileIdentity()] = append(reports[fh.VersionedFileIdentity()], diagnostics...)
	return nil
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

// hasUndeclaredErrors returns true if a package has a type error
// about an undeclared symbol.
func hasUndeclaredErrors(pkg Package) bool {
	for _, err := range pkg.GetErrors() {
		if err.Kind != TypeError {
			continue
		}
		if strings.Contains(err.Message, "undeclared name:") {
			return true
		}
	}
	return false
}

func isConvenienceAnalyzer(o *Options, category string) bool {
	for _, a := range DefaultOptions().ConvenienceAnalyzers {
		if category == a.Analyzer.Name {
			return true
		}
	}
	return false
}
