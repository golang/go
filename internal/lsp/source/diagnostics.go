// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"strings"

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
}

type DiagnosticSeverity int

const (
	SeverityWarning DiagnosticSeverity = iota
	SeverityError
)

func Diagnostics(ctx context.Context, view View, f File, disabledAnalyses map[string]struct{}) (map[span.URI][]Diagnostic, string, error) {
	ctx, done := trace.StartSpan(ctx, "source.Diagnostics", telemetry.File.Of(f.URI()))
	defer done()

	_, cphs, err := view.CheckPackageHandles(ctx, f)
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
		clearReports(view, reports, packagesErrorSpan(err).URI())
	}

	// Run diagnostics for the package that this URI belongs to.
	if !diagnostics(ctx, view, pkg, reports) {
		// If we don't have any list, parse, or type errors, run analyses.
		if err := analyses(ctx, view, cph, disabledAnalyses, reports); err != nil {
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
		spn := packagesErrorSpan(err)
		diag := &Diagnostic{
			URI:      spn.URI(),
			Message:  err.Msg,
			Source:   "LSP",
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
		case packages.TypeError:
			set.typeErrors = append(set.typeErrors, diag)
		default:
			set.listErrors = append(set.listErrors, diag)
		}
		rng, err := spanToRange(ctx, view, pkg, spn, err.Kind == packages.TypeError)
		if err != nil {
			log.Error(ctx, "failed to convert span to range", err)
			continue
		}
		diag.Range = rng
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

// spanToRange converts a span.Span to a protocol.Range,
// assuming that the span belongs to the package whose diagnostics are being computed.
func spanToRange(ctx context.Context, view View, pkg Package, spn span.Span, isTypeError bool) (protocol.Range, error) {
	ph, err := pkg.File(spn.URI())
	if err != nil {
		return protocol.Range{}, err
	}
	_, m, _, err := ph.Cached(ctx)
	if err != nil {
		return protocol.Range{}, err
	}
	data, _, err := ph.File().Read(ctx)
	if err != nil {
		return protocol.Range{}, err
	}
	// Try to get a range for the diagnostic.
	// TODO: Don't just limit ranges to type errors.
	if spn.IsPoint() && isTypeError {
		if s, err := spn.WithOffset(m.Converter); err == nil {
			start := s.Start()
			offset := start.Offset()
			if width := bytes.IndexAny(data[offset:], " \n,():;[]"); width > 0 {
				spn = span.New(spn.URI(), start, span.NewPoint(start.Line(), start.Column()+width, offset+width))
			}
		}
	}
	return m.Range(spn)
}

func analyses(ctx context.Context, view View, cph CheckPackageHandle, disabledAnalyses map[string]struct{}, reports map[span.URI][]Diagnostic) error {
	// Type checking and parsing succeeded. Run analyses.
	if err := runAnalyses(ctx, view, cph, disabledAnalyses, func(a *analysis.Analyzer, diag analysis.Diagnostic) error {
		diagnostic, err := toDiagnostic(ctx, view, diag, a.Name)
		if err != nil {
			return err
		}
		addReport(view, reports, diagnostic.URI, diagnostic)
		return nil
	}); err != nil {
		return err
	}
	return nil
}

func toDiagnostic(ctx context.Context, view View, diag analysis.Diagnostic, category string) (Diagnostic, error) {
	r := span.NewRange(view.Session().Cache().FileSet(), diag.Pos, diag.End)
	spn, err := r.Span()
	if err != nil {
		// The diagnostic has an invalid position, so we don't have a valid span.
		return Diagnostic{}, err
	}
	if diag.Category != "" {
		category += "." + category
	}
	f, err := view.GetFile(ctx, spn.URI())
	if err != nil {
		return Diagnostic{}, err
	}
	// If the package has changed since these diagnostics were computed,
	// this may be incorrect. Should the package be associated with the diagnostic?
	_, cphs, err := view.CheckPackageHandles(ctx, f)
	if err != nil {
		return Diagnostic{}, err
	}
	cph, err := NarrowestCheckPackageHandle(cphs)
	if err != nil {
		return Diagnostic{}, err
	}
	pkg, err := cph.Cached(ctx)
	if err != nil {
		return Diagnostic{}, err
	}
	rng, err := spanToRange(ctx, view, pkg, spn, false)
	if err != nil {
		return Diagnostic{}, err
	}
	fixes, err := suggestedFixes(ctx, view, pkg, diag)
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
	return Diagnostic{
		URI:            spn.URI(),
		Range:          rng,
		Source:         category,
		Message:        diag.Message,
		Severity:       protocol.SeverityWarning,
		SuggestedFixes: fixes,
		Tags:           tags,
	}, nil
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

func packagesErrorSpan(err packages.Error) span.Span {
	if err.Pos == "" {
		return parseDiagnosticMessage(err.Msg)
	}
	return span.Parse(err.Pos)
}

// parseDiagnosticMessage attempts to parse a standard `go list` error message
// by stripping off the trailing error message.
//
// It works only on errors whose message is prefixed by colon,
// followed by a space (": "). For example:
//
//   attributes.go:13:1: expected 'package', found 'type'
//
func parseDiagnosticMessage(input string) span.Span {
	input = strings.TrimSpace(input)
	msgIndex := strings.Index(input, ": ")
	if msgIndex < 0 {
		return span.Parse(input)
	}
	return span.Parse(input[:msgIndex])
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

func runAnalyses(ctx context.Context, view View, cph CheckPackageHandle, disabledAnalyses map[string]struct{}, report func(a *analysis.Analyzer, diag analysis.Diagnostic) error) error {
	var analyzers []*analysis.Analyzer
	for _, a := range view.Options().Analyzers {
		if _, ok := disabledAnalyses[a.Name]; ok {
			continue
		}
		analyzers = append(analyzers, a)
	}

	roots, err := analyze(ctx, view, []CheckPackageHandle{cph}, analyzers)
	if err != nil {
		return err
	}

	// Report diagnostics and errors from root analyzers.
	for _, r := range roots {
		var sdiags []Diagnostic
		for _, diag := range r.diagnostics {
			if r.err != nil {
				// TODO(matloob): This isn't quite right: we might return a failed prerequisites error,
				// which isn't super useful...
				return r.err
			}
			if err := report(r.Analyzer, diag); err != nil {
				return err
			}
			sdiag, err := toDiagnostic(ctx, view, diag, r.Analyzer.Name)
			if err != nil {
				return err
			}
			sdiags = append(sdiags, sdiag)
		}
		pkg, err := cph.Check(ctx)
		if err != nil {
			return err
		}
		pkg.SetDiagnostics(r.Analyzer, sdiags)
	}
	return nil
}
