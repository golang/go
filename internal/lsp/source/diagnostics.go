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
	"golang.org/x/tools/go/analysis/passes/asmdecl"
	"golang.org/x/tools/go/analysis/passes/assign"
	"golang.org/x/tools/go/analysis/passes/atomic"
	"golang.org/x/tools/go/analysis/passes/atomicalign"
	"golang.org/x/tools/go/analysis/passes/bools"
	"golang.org/x/tools/go/analysis/passes/buildtag"
	"golang.org/x/tools/go/analysis/passes/cgocall"
	"golang.org/x/tools/go/analysis/passes/composite"
	"golang.org/x/tools/go/analysis/passes/copylock"
	"golang.org/x/tools/go/analysis/passes/httpresponse"
	"golang.org/x/tools/go/analysis/passes/loopclosure"
	"golang.org/x/tools/go/analysis/passes/lostcancel"
	"golang.org/x/tools/go/analysis/passes/nilfunc"
	"golang.org/x/tools/go/analysis/passes/printf"
	"golang.org/x/tools/go/analysis/passes/shift"
	"golang.org/x/tools/go/analysis/passes/stdmethods"
	"golang.org/x/tools/go/analysis/passes/structtag"
	"golang.org/x/tools/go/analysis/passes/tests"
	"golang.org/x/tools/go/analysis/passes/unmarshal"
	"golang.org/x/tools/go/analysis/passes/unreachable"
	"golang.org/x/tools/go/analysis/passes/unsafeptr"
	"golang.org/x/tools/go/analysis/passes/unusedresult"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/lsp/telemetry/log"
	"golang.org/x/tools/internal/lsp/telemetry/trace"
	"golang.org/x/tools/internal/span"
)

type Diagnostic struct {
	span.Span
	Message  string
	Source   string
	Severity DiagnosticSeverity

	SuggestedFixes []SuggestedFixes
}

type SuggestedFixes struct {
	Title string
	Edits []TextEdit
}

type DiagnosticSeverity int

const (
	SeverityWarning DiagnosticSeverity = iota
	SeverityError
)

func Diagnostics(ctx context.Context, view View, f GoFile, disabledAnalyses map[string]struct{}) (map[span.URI][]Diagnostic, error) {
	ctx, done := trace.StartSpan(ctx, "source.Diagnostics", telemetry.File.Of(f.URI()))
	defer done()
	pkg := f.GetPackage(ctx)
	if pkg == nil {
		return singleDiagnostic(f.URI(), "%s is not part of a package", f.URI()), nil
	}
	// Prepare the reports we will send for the files in this package.
	reports := make(map[span.URI][]Diagnostic)
	for _, filename := range pkg.GetFilenames() {
		clearReports(view, reports, span.FileURI(filename))
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
		if err := analyses(ctx, view, pkg, disabledAnalyses, reports); err != nil {
			log.Error(ctx, "failed to run analyses", err, telemetry.File)
		}
	}
	// Updates to the diagnostics for this package may need to be propagated.
	revDeps := f.GetActiveReverseDeps(ctx)
	for _, f := range revDeps {
		pkg := f.GetPackage(ctx)
		if pkg == nil {
			continue
		}
		for _, filename := range pkg.GetFilenames() {
			clearReports(view, reports, span.FileURI(filename))
		}
		diagnostics(ctx, view, pkg, reports)
	}
	return reports, nil
}

type diagnosticSet struct {
	listErrors, parseErrors, typeErrors []Diagnostic
}

func diagnostics(ctx context.Context, v View, pkg Package, reports map[span.URI][]Diagnostic) bool {
	ctx, done := trace.StartSpan(ctx, "source.diagnostics", telemetry.Package.Of(pkg.ID()))
	defer done()
	diagSets := make(map[span.URI]*diagnosticSet)
	for _, err := range pkg.GetErrors() {
		diag := Diagnostic{
			Span:     packagesErrorSpan(err),
			Message:  err.Msg,
			Source:   "LSP",
			Severity: SeverityError,
		}
		set, ok := diagSets[diag.Span.URI()]
		if !ok {
			set = &diagnosticSet{}
			diagSets[diag.Span.URI()] = set
		}
		switch err.Kind {
		case packages.ParseError:
			set.parseErrors = append(set.parseErrors, diag)
		case packages.TypeError:
			if diag.Span.IsPoint() {
				diag.Span = pointToSpan(ctx, v, diag.Span)
			}
			set.typeErrors = append(set.typeErrors, diag)
		default:
			set.listErrors = append(set.listErrors, diag)
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
				reports[uri] = append(reports[uri], diag)
			}
		}
	}
	return nonEmptyDiagnostics
}

func analyses(ctx context.Context, v View, pkg Package, disabledAnalyses map[string]struct{}, reports map[span.URI][]Diagnostic) error {
	// Type checking and parsing succeeded. Run analyses.
	if err := runAnalyses(ctx, v, pkg, disabledAnalyses, func(a *analysis.Analyzer, diag analysis.Diagnostic) error {
		diagnostic, err := toDiagnostic(a, v, diag)
		if err != nil {
			return err
		}
		addReport(v, reports, diagnostic.Span.URI(), diagnostic)
		return nil
	}); err != nil {
		return err
	}
	return nil
}

func toDiagnostic(a *analysis.Analyzer, v View, diag analysis.Diagnostic) (Diagnostic, error) {
	r := span.NewRange(v.Session().Cache().FileSet(), diag.Pos, diag.End)
	s, err := r.Span()
	if err != nil {
		// The diagnostic has an invalid position, so we don't have a valid span.
		return Diagnostic{}, err
	}
	category := a.Name
	if diag.Category != "" {
		category += "." + category
	}
	ca, err := getCodeActions(v.Session().Cache().FileSet(), diag)
	if err != nil {
		return Diagnostic{}, err
	}
	return Diagnostic{
		Source:         category,
		Span:           s,
		Message:        diag.Message,
		Severity:       SeverityWarning,
		SuggestedFixes: ca,
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
	reports[uri] = append(reports[uri], diagnostic)
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

func pointToSpan(ctx context.Context, view View, spn span.Span) span.Span {
	f, err := view.GetFile(ctx, spn.URI())
	ctx = telemetry.File.With(ctx, spn.URI())
	if err != nil {
		log.Error(ctx, "could not find file for diagnostic", nil, telemetry.File)
		return spn
	}
	diagFile, ok := f.(GoFile)
	if !ok {
		log.Error(ctx, "not a Go file", nil, telemetry.File)
		return spn
	}
	tok, err := diagFile.GetToken(ctx)
	if err != nil {
		log.Error(ctx, "could not find token.File for diagnostic", err, telemetry.File)
		return spn
	}
	data, _, err := diagFile.Handle(ctx).Read(ctx)
	if err != nil {
		log.Error(ctx, "could not find content for diagnostic", err, telemetry.File)
		return spn
	}
	c := span.NewTokenConverter(diagFile.FileSet(), tok)
	s, err := spn.WithOffset(c)
	//we just don't bother producing an error if this failed
	if err != nil {
		log.Error(ctx, "invalid span for diagnostic", err, telemetry.File)
		return spn
	}
	start := s.Start()
	offset := start.Offset()
	width := bytes.IndexAny(data[offset:], " \n,():;[]")
	if width <= 0 {
		return spn
	}
	return span.New(spn.URI(), start, span.NewPoint(start.Line(), start.Column()+width, offset+width))
}

func singleDiagnostic(uri span.URI, format string, a ...interface{}) map[span.URI][]Diagnostic {
	return map[span.URI][]Diagnostic{
		uri: []Diagnostic{{
			Source:   "LSP",
			Span:     span.New(uri, span.Point{}, span.Point{}),
			Message:  fmt.Sprintf(format, a...),
			Severity: SeverityError,
		}},
	}
}

var Analyzers = []*analysis.Analyzer{
	// The traditional vet suite:
	asmdecl.Analyzer,
	assign.Analyzer,
	atomic.Analyzer,
	atomicalign.Analyzer,
	bools.Analyzer,
	buildtag.Analyzer,
	cgocall.Analyzer,
	composite.Analyzer,
	copylock.Analyzer,
	httpresponse.Analyzer,
	loopclosure.Analyzer,
	lostcancel.Analyzer,
	nilfunc.Analyzer,
	printf.Analyzer,
	shift.Analyzer,
	stdmethods.Analyzer,
	structtag.Analyzer,
	tests.Analyzer,
	unmarshal.Analyzer,
	unreachable.Analyzer,
	unsafeptr.Analyzer,
	unusedresult.Analyzer,
}

func runAnalyses(ctx context.Context, v View, pkg Package, disabledAnalyses map[string]struct{}, report func(a *analysis.Analyzer, diag analysis.Diagnostic) error) error {
	var analyzers []*analysis.Analyzer
	for _, a := range Analyzers {
		if _, ok := disabledAnalyses[a.Name]; ok {
			continue
		}
		analyzers = append(analyzers, a)
	}

	roots, err := analyze(ctx, v, []Package{pkg}, analyzers)
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
			sdiag, err := toDiagnostic(r.Analyzer, v, diag)
			if err != nil {
				return err
			}
			sdiags = append(sdiags, sdiag)
		}
		pkg.SetDiagnostics(sdiags)
	}
	return nil
}
