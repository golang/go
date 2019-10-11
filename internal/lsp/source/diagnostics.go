// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
)

type Diagnostic struct {
	URI      span.URI
	Range    protocol.Range
	Message  string
	Source   string
	Severity protocol.DiagnosticSeverity
	Tags     []protocol.DiagnosticTag

	SuggestedFixes []SuggestedFix
	Related        []RelatedInformation
}

type RelatedInformation struct {
	URI     span.URI
	Range   protocol.Range
	Message string
}

type DiagnosticSeverity int

const (
	SeverityWarning DiagnosticSeverity = iota
	SeverityError
)

func Diagnostics(ctx context.Context, view View, f File, disabledAnalyses map[string]struct{}) (map[span.URI][]Diagnostic, string, error) {
	ctx, done := trace.StartSpan(ctx, "source.Diagnostics", telemetry.File.Of(f.URI()))
	defer done()

	snapshot, cphs, err := view.CheckPackageHandles(ctx, f)
	if err != nil {
		return nil, "", err
	}
	cph, err := WidestCheckPackageHandle(cphs)
	if err != nil {
		return nil, "", err
	}

	// If we are missing dependencies, it may because the user's workspace is
	// not correctly configured. Report errors, if possible.
	var warningMsg string
	if len(cph.MissingDependencies()) > 0 {
		warningMsg, err = checkCommonErrors(ctx, view, f.URI())
		if err != nil {
			log.Error(ctx, "error checking common errors", err, telemetry.File.Of(f.URI))
		}
	}
	pkg, err := cph.Check(ctx)
	if err != nil {
		log.Error(ctx, "no package for file", err)
		return singleDiagnostic(f.URI(), "%s is not part of a package", f.URI()), "", nil
	}

	// Prepare the reports we will send for the files in this package.
	reports := make(map[span.URI][]Diagnostic)
	for _, fh := range pkg.Files() {
		clearReports(view, reports, fh.File().Identity().URI)
	}

	// Prepare any additional reports for the errors in this package.
	for _, err := range pkg.GetErrors() {
		if err.Kind != packages.ListError {
			continue
		}
		clearReports(view, reports, err.URI)
	}

	// Run diagnostics for the package that this URI belongs to.
	if !diagnostics(ctx, view, pkg, reports) {
		// If we don't have any list, parse, or type errors, run analyses.
		if err := analyses(ctx, snapshot, cph, disabledAnalyses, reports); err != nil {
			log.Error(ctx, "failed to run analyses", err, telemetry.File.Of(f.URI()))
		}
	}
	// Updates to the diagnostics for this package may need to be propagated.
	revDeps := view.GetActiveReverseDeps(ctx, f)
	for _, cph := range revDeps {
		pkg, err := cph.Check(ctx)
		if err != nil {
			return nil, warningMsg, err
		}
		for _, fh := range pkg.Files() {
			clearReports(view, reports, fh.File().Identity().URI)
		}
		diagnostics(ctx, view, pkg, reports)
	}
	return reports, warningMsg, nil
}

type diagnosticSet struct {
	listErrors, parseErrors, typeErrors []*Diagnostic
}

func diagnostics(ctx context.Context, view View, pkg Package, reports map[span.URI][]Diagnostic) bool {
	ctx, done := trace.StartSpan(ctx, "source.diagnostics", telemetry.Package.Of(pkg.ID()))
	defer done()

	diagSets := make(map[span.URI]*diagnosticSet)
	for _, err := range pkg.GetErrors() {
		diag := &Diagnostic{
			URI:      err.URI,
			Message:  err.Msg,
			Range:    err.Range,
			Severity: protocol.SeverityError,
		}
		set, ok := diagSets[diag.URI]
		if !ok {
			set = &diagnosticSet{}
			diagSets[diag.URI] = set
		}
		switch err.Kind {
		case packages.ParseError:
			set.parseErrors = append(set.parseErrors, diag)
			diag.Source = "syntax"
		case packages.TypeError:
			set.typeErrors = append(set.typeErrors, diag)
			diag.Source = "compiler"
		default:
			set.listErrors = append(set.listErrors, diag)
			diag.Source = "go list"
		}
	}
	var nonEmptyDiagnostics bool // track if we actually send non-empty diagnostics
	for uri, set := range diagSets {
		// Don't report type errors if there are parse errors or list errors.
		diags := set.typeErrors
		if len(set.parseErrors) > 0 {
			diags = set.parseErrors
		} else if len(set.listErrors) > 0 {
			diags = set.listErrors
		}
		if len(diags) > 0 {
			nonEmptyDiagnostics = true
		}
		for _, diag := range diags {
			if _, ok := reports[uri]; ok {
				reports[uri] = append(reports[uri], *diag)
			}
		}
	}
	return nonEmptyDiagnostics
}

func analyses(ctx context.Context, snapshot Snapshot, cph CheckPackageHandle, disabledAnalyses map[string]struct{}, reports map[span.URI][]Diagnostic) error {
	// Type checking and parsing succeeded. Run analyses.
	if err := runAnalyses(ctx, snapshot, cph, disabledAnalyses, func(diags []*analysis.Diagnostic, a *analysis.Analyzer) error {
		for _, diag := range diags {
			diagnostic, err := toDiagnostic(ctx, snapshot.View(), diag, a.Name)
			if err != nil {
				return err
			}
			addReport(snapshot.View(), reports, diagnostic.URI, diagnostic)
		}
		return nil
	}); err != nil {
		return err
	}
	return nil
}

func packageForSpan(ctx context.Context, view View, spn span.Span) (Package, error) {
	f, err := view.GetFile(ctx, spn.URI())
	if err != nil {
		return nil, err
	}
	// If the package has changed since these diagnostics were computed,
	// this may be incorrect. Should the package be associated with the diagnostic?
	_, cphs, err := view.CheckPackageHandles(ctx, f)
	if err != nil {
		return nil, err
	}
	cph, err := NarrowestCheckPackageHandle(cphs)
	if err != nil {
		return nil, err
	}
	return cph.Cached(ctx)
}

func toDiagnostic(ctx context.Context, view View, diag *analysis.Diagnostic, category string) (Diagnostic, error) {
	spn, err := span.NewRange(view.Session().Cache().FileSet(), diag.Pos, diag.End).Span()
	if err != nil {
		return Diagnostic{}, err
	}
	pkg, err := packageForSpan(ctx, view, spn)
	if err != nil {
		return Diagnostic{}, err
	}
	ph, err := pkg.File(spn.URI())
	if err != nil {
		return Diagnostic{}, err
	}
	_, m, _, err := ph.Cached(ctx)
	if err != nil {
		return Diagnostic{}, err
	}
	rng, err := m.Range(spn)
	if err != nil {
		return Diagnostic{}, err
	}
	fixes, err := suggestedFixes(ctx, view, pkg, diag)
	if err != nil {
		return Diagnostic{}, err
	}

	related, err := relatedInformation(ctx, view, diag)
	if err != nil {
		return Diagnostic{}, err
	}

	// This is a bit of a hack, but clients > 3.15 will be able to grey out unnecessary code.
	// If we are deleting code as part of all of our suggested fixes, assume that this is dead code.
	// TODO(golang/go/#34508): Return these codes from the diagnostics themselves.
	var tags []protocol.DiagnosticTag
	if onlyDeletions(fixes) {
		tags = append(tags, protocol.Unnecessary)
	}
	if diag.Category != "" {
		category += "." + diag.Category
	}
	return Diagnostic{
		URI:            spn.URI(),
		Range:          rng,
		Source:         category,
		Message:        diag.Message,
		Severity:       protocol.SeverityWarning,
		SuggestedFixes: fixes,
		Tags:           tags,
		Related:        related,
	}, nil
}

func relatedInformation(ctx context.Context, view View, diag *analysis.Diagnostic) ([]RelatedInformation, error) {
	var out []RelatedInformation
	for _, related := range diag.Related {
		r := span.NewRange(view.Session().Cache().FileSet(), related.Pos, related.End)
		spn, err := r.Span()
		if err != nil {
			return nil, err
		}
		pkg, err := packageForSpan(ctx, view, spn)
		if err != nil {
			return nil, err
		}
		rng, err := spanToRange(ctx, view, pkg, spn, false)
		if err != nil {
			return nil, err
		}

		out = append(out, RelatedInformation{
			URI:     spn.URI(),
			Range:   rng,
			Message: related.Message,
		})
	}
	return out, nil
}

func clearReports(v View, reports map[span.URI][]Diagnostic, uri span.URI) {
	if v.Ignore(uri) {
		return
	}
	reports[uri] = []Diagnostic{}
}

func addReport(v View, reports map[span.URI][]Diagnostic, uri span.URI, diagnostic Diagnostic) {
	if v.Ignore(uri) {
		return
	}
	if _, ok := reports[uri]; ok {
		reports[uri] = append(reports[uri], diagnostic)
	}
}

func singleDiagnostic(uri span.URI, format string, a ...interface{}) map[span.URI][]Diagnostic {
	return map[span.URI][]Diagnostic{
		uri: []Diagnostic{{
			Source:   "LSP",
			URI:      uri,
			Range:    protocol.Range{},
			Message:  fmt.Sprintf(format, a...),
			Severity: protocol.SeverityError,
		}},
	}
}

func runAnalyses(ctx context.Context, snapshot Snapshot, cph CheckPackageHandle, disabledAnalyses map[string]struct{}, report func(diag []*analysis.Diagnostic, a *analysis.Analyzer) error) error {
	var analyzers []*analysis.Analyzer
	for _, a := range snapshot.View().Options().Analyzers {
		if _, ok := disabledAnalyses[a.Name]; ok {
			continue
		}
		analyzers = append(analyzers, a)
	}

	diagnostics, err := snapshot.Analyze(ctx, cph.ID(), analyzers)
	if err != nil {
		return err
	}

	// Report diagnostics and errors from root analyzers.
	var sdiags []Diagnostic
	for a, diags := range diagnostics {
		if err := report(diags, a); err != nil {
			return err
		}
		for _, diag := range diags {
			sdiag, err := toDiagnostic(ctx, snapshot.View(), diag, a.Name)
			if err != nil {
				return err
			}
			sdiags = append(sdiags, sdiag)
		}
		pkg, err := cph.Check(ctx)
		if err != nil {
			return err
		}
		pkg.SetDiagnostics(a, sdiags)
	}
	return nil
}
