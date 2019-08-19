// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
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
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

type Diagnostic struct {
	URI      span.URI
	Range    protocol.Range
	Message  string
	Source   string
	Severity DiagnosticSeverity

	SuggestedFixes []SuggestedFixes
}

type SuggestedFixes struct {
	Title string
	Edits []diff.TextEdit
}

type DiagnosticSeverity int

const (
	SeverityWarning DiagnosticSeverity = iota
	SeverityError
)

func Diagnostics(ctx context.Context, view View, f GoFile, disabledAnalyses map[string]struct{}) (map[span.URI][]Diagnostic, error) {
	ctx, done := trace.StartSpan(ctx, "source.Diagnostics", telemetry.File.Of(f.URI()))
	defer done()

	cph, err := f.GetCheckPackageHandle(ctx)
	if err != nil {
		return nil, err
	}
	pkg, err := cph.Check(ctx)
	if err != nil {
		log.Error(ctx, "no package for file", err)
		return singleDiagnostic(f.URI(), "%s is not part of a package", f.URI()), nil
	}
	// Prepare the reports we will send for the files in this package.
	reports := make(map[span.URI][]Diagnostic)
	for _, fh := range pkg.GetHandles() {
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
	revDeps := f.GetActiveReverseDeps(ctx)
	for _, f := range revDeps {
		pkg, err := f.GetPackage(ctx)
		if err != nil {
			return nil, err
		}
		for _, fh := range pkg.GetHandles() {
			clearReports(view, reports, fh.File().Identity().URI)
		}
		diagnostics(ctx, view, pkg, reports)
	}
	return reports, nil
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
			Severity: SeverityError,
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
	var (
		fh   FileHandle
		file *ast.File
		err  error
	)
	for _, ph := range pkg.GetHandles() {
		if ph.File().Identity().URI == spn.URI() {
			fh = ph.File()
			file, err = ph.Cached(ctx)
		}
	}
	if file == nil {
		return protocol.Range{}, err
	}
	fset := view.Session().Cache().FileSet()
	tok := fset.File(file.Pos())
	if tok == nil {
		return protocol.Range{}, errors.Errorf("no token.File for %s", spn.URI())
	}
	data, _, err := fh.Read(ctx)
	if err != nil {
		return protocol.Range{}, err
	}
	uri := fh.Identity().URI
	m := protocol.NewColumnMapper(uri, uri.Filename(), fset, tok, data)

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
	ca, err := getCodeActions(view.Session().Cache().FileSet(), diag)
	if err != nil {
		return Diagnostic{}, err
	}
	f, err := view.GetFile(ctx, spn.URI())
	if err != nil {
		return Diagnostic{}, err
	}
	gof, ok := f.(GoFile)
	if !ok {
		return Diagnostic{}, errors.Errorf("%s is not a Go file", f.URI())
	}
	// If the package has changed since these diagnostics were computed,
	// this may be incorrect. Should the package be associated with the diagnostic?
	pkg, err := gof.GetCachedPackage(ctx)
	if err != nil {
		return Diagnostic{}, err
	}
	rng, err := spanToRange(ctx, view, pkg, spn, false)
	if err != nil {
		return Diagnostic{}, err
	}
	return Diagnostic{
		URI:            spn.URI(),
		Range:          rng,
		Source:         category,
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

func runAnalyses(ctx context.Context, view View, cph CheckPackageHandle, disabledAnalyses map[string]struct{}, report func(a *analysis.Analyzer, diag analysis.Diagnostic) error) error {
	var analyzers []*analysis.Analyzer
	for _, a := range Analyzers {
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
		pkg.SetDiagnostics(sdiags)
	}
	return nil
}
