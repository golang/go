// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"

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
	pkg := f.GetPackage(ctx)
	if pkg == nil {
		return singleDiagnostic(uri, "%s is not part of a package", uri), nil
	}
	// Prepare the reports we will send for this package.
	reports := make(map[span.URI][]Diagnostic)
	for _, filename := range pkg.GetFilenames() {
		reports[span.FileURI(filename)] = []Diagnostic{}
	}
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
		spn := span.Parse(diag.Pos)
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
	if len(diags) > 0 {
		return reports, nil
	}
	// Type checking and parsing succeeded. Run analyses.
	if err := runAnalyses(ctx, v, pkg, func(a *analysis.Analyzer, diag analysis.Diagnostic) error {
		r := span.NewRange(v.FileSet(), diag.Pos, 0)
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
		return singleDiagnostic(uri, "unable to run analyses for %s: %v", uri, err), nil
	}

	return reports, nil
}

func pointToSpan(ctx context.Context, v View, spn span.Span) span.Span {
	// Don't set a range if it's anything other than a type error.
	diagFile, err := v.GetFile(ctx, spn.URI())
	if err != nil {
		v.Logger().Errorf(ctx, "Could find file for diagnostic: %v", spn.URI())
		return spn
	}
	tok := diagFile.GetToken(ctx)
	if tok == nil {
		v.Logger().Errorf(ctx, "Could not find tokens for diagnostic: %v", spn.URI())
		return spn
	}
	content := diagFile.GetContent(ctx)
	if content == nil {
		v.Logger().Errorf(ctx, "Could not find content for diagnostic: %v", spn.URI())
		return spn
	}
	c := span.NewTokenConverter(diagFile.GetFileSet(ctx), tok)
	s, err := spn.WithOffset(c)
	//we just don't bother producing an error if this failed
	if err != nil {
		v.Logger().Errorf(ctx, "invalid span for diagnostic: %v: %v", spn.URI(), err)
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
