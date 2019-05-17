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
	"golang.org/x/tools/internal/span"
)

type Diagnostic struct {
	span.Span
	Message  string
	Source   string
	Severity DiagnosticSeverity
}

type DiagnosticSeverity int

const (
	SeverityWarning DiagnosticSeverity = iota
	SeverityError
)

func Diagnostics(ctx context.Context, v View, uri span.URI) (map[span.URI][]Diagnostic, error) {
	f, err := v.GetFile(ctx, uri)
	if err != nil {
		return singleDiagnostic(uri, "no file found for %s", uri), nil
	}
	gof, ok := f.(GoFile)
	if !ok {
		return singleDiagnostic(uri, "%s is not a go file", uri), nil
	}
	pkg := gof.GetPackage(ctx)
	if pkg == nil {
		return singleDiagnostic(uri, "%s is not part of a package", uri), nil
	}
	// Prepare the reports we will send for this package.
	reports := make(map[span.URI][]Diagnostic)
	for _, filename := range pkg.GetFilenames() {
		uri := span.FileURI(filename)
		if v.Ignore(uri) {
			continue
		}
		reports[uri] = []Diagnostic{}
	}

	// Prepare reports for package errors
	for _, pkgErr := range pkg.GetErrors() {
		reports[packageErrorSpan(pkgErr).URI()] = []Diagnostic{}
	}

	// Run diagnostics for the package that this URI belongs to.
	if !diagnostics(ctx, v, pkg, reports) {
		// If we don't have any list, parse, or type errors, run analyses.
		if err := analyses(ctx, v, pkg, reports); err != nil {
			return singleDiagnostic(uri, "failed to run analyses for %s: %v", uri, err), nil
		}
	}
	// Updates to the diagnostics for this package may need to be propagated.
	for _, f := range gof.GetActiveReverseDeps(ctx) {
		pkg := f.GetPackage(ctx)
		if pkg == nil {
			continue
		}
		for _, filename := range pkg.GetFilenames() {
			reports[span.FileURI(filename)] = []Diagnostic{}
		}
		diagnostics(ctx, v, pkg, reports)
	}
	return reports, nil
}

func diagnostics(ctx context.Context, v View, pkg Package, reports map[span.URI][]Diagnostic) bool {
	var listErrors, parseErrors, typeErrors []packages.Error
	for _, err := range pkg.GetErrors() {
		switch err.Kind {
		case packages.ParseError:
			parseErrors = append(parseErrors, err)
		case packages.TypeError:
			typeErrors = append(typeErrors, err)
		default:
			listErrors = append(listErrors, err)
		}
	}
	// Don't report type errors if there are parse errors or list errors.
	diags := typeErrors
	if len(parseErrors) > 0 {
		diags = parseErrors
	} else if len(listErrors) > 0 {
		diags = listErrors
	}
	for _, diag := range diags {
		spn := packageErrorSpan(diag)
		if spn.IsPoint() && diag.Kind == packages.TypeError {
			spn = pointToSpan(ctx, v, spn)
		}
		diagnostic := Diagnostic{
			Source:   "LSP",
			Span:     spn,
			Message:  diag.Msg,
			Severity: SeverityError,
		}
		if _, ok := reports[spn.URI()]; ok {
			reports[spn.URI()] = append(reports[spn.URI()], diagnostic)
		}
	}
	// Returns true if we've sent non-empty diagnostics.
	return len(diags) != 0
}

func analyses(ctx context.Context, v View, pkg Package, reports map[span.URI][]Diagnostic) error {
	// Type checking and parsing succeeded. Run analyses.
	if err := runAnalyses(ctx, v, pkg, func(a *analysis.Analyzer, diag analysis.Diagnostic) error {
		r := span.NewRange(v.Session().Cache().FileSet(), diag.Pos, 0)
		s, err := r.Span()
		if err != nil {
			// The diagnostic has an invalid position, so we don't have a valid span.
			return err
		}
		category := a.Name
		if diag.Category != "" {
			category += "." + category
		}
		reports[s.URI()] = append(reports[s.URI()], Diagnostic{
			Source:   category,
			Span:     s,
			Message:  diag.Message,
			Severity: SeverityWarning,
		})
		return nil
	}); err != nil {
		return err
	}
	return nil
}

// parseDiagnosticMessage attempts to parse a standard error message by stripping off the trailing error message.
// Works only on errors where the message is prefixed by ": ".
// e.g.:
//   attributes.go:13:1: expected 'package', found 'type'
func parseDiagnosticMessage(input string) span.Span {
	input = strings.TrimSpace(input)

	msgIndex := strings.Index(input, ": ")
	if msgIndex < 0 {
		return span.Parse(input)
	}

	return span.Parse(input[:msgIndex])
}

func packageErrorSpan(pkgErr packages.Error) span.Span {
	if pkgErr.Pos == "" {
		return parseDiagnosticMessage(pkgErr.Msg)
	}

	return span.Parse(pkgErr.Pos)
}

func pointToSpan(ctx context.Context, v View, spn span.Span) span.Span {
	// Don't set a range if it's anything other than a type error.
	f, err := v.GetFile(ctx, spn.URI())
	if err != nil {
		v.Session().Logger().Errorf(ctx, "Could find file for diagnostic: %v", spn.URI())
		return spn
	}
	diagFile, ok := f.(GoFile)
	if !ok {
		v.Session().Logger().Errorf(ctx, "Not a go file: %v", spn.URI())
		return spn
	}
	tok := diagFile.GetToken(ctx)
	if tok == nil {
		v.Session().Logger().Errorf(ctx, "Could not find tokens for diagnostic: %v", spn.URI())
		return spn
	}
	content := diagFile.GetContent(ctx)
	if content == nil {
		v.Session().Logger().Errorf(ctx, "Could not find content for diagnostic: %v", spn.URI())
		return spn
	}
	c := span.NewTokenConverter(diagFile.FileSet(), tok)
	s, err := spn.WithOffset(c)
	//we just don't bother producing an error if this failed
	if err != nil {
		v.Session().Logger().Errorf(ctx, "invalid span for diagnostic: %v: %v", spn.URI(), err)
		return spn
	}
	start := s.Start()
	offset := start.Offset()
	width := bytes.IndexAny(content[offset:], " \n,():;[]")
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

func runAnalyses(ctx context.Context, v View, pkg Package, report func(a *analysis.Analyzer, diag analysis.Diagnostic) error) error {
	// The traditional vet suite:
	analyzers := []*analysis.Analyzer{
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

	roots, err := analyze(ctx, v, []Package{pkg}, analyzers)
	if err != nil {
		return err
	}

	// Report diagnostics and errors from root analyzers.
	for _, r := range roots {
		for _, diag := range r.diagnostics {
			if r.err != nil {
				// TODO(matloob): This isn't quite right: we might return a failed prerequisites error,
				// which isn't super useful...
				return r.err
			}
			if err := report(r.Analyzer, diag); err != nil {
				return err
			}
		}
	}
	return nil
}
