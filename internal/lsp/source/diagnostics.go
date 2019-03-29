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
			// Don't set a range if it's anything other than a type error.
			if diagFile, err := v.GetFile(ctx, spn.URI()); err == nil {
				tok := diagFile.GetToken(ctx)
				if tok == nil {
					v.Logger().Errorf(ctx, "Could not matching tokens for diagnostic: %v", diagFile.URI())
					continue
				}
				content := diagFile.GetContent(ctx)
				c := span.NewTokenConverter(diagFile.GetFileSet(ctx), tok)
				s, err := spn.WithOffset(c)
				//we just don't bother producing an error if this failed
				if err == nil {
					start := s.Start()
					offset := start.Offset()
					if l := bytes.IndexAny(content[offset:], " \n,():;[]"); l > 0 {
						spn = span.New(spn.URI(), start, span.NewPoint(start.Line(), start.Column()+l, offset+l))
					}
				}
			}
		}
		diagnostic := Diagnostic{
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
	runAnalyses(ctx, v, pkg, func(a *analysis.Analyzer, diag analysis.Diagnostic) {
		r := span.NewRange(v.FileSet(), diag.Pos, 0)
		s, err := r.Span()
		if err != nil {
			//TODO: we could not process the diag.Pos, and thus have no valid span
			//we don't have anywhere to put this error though
			v.Logger().Errorf(ctx, "%v", err)
		}
		category := a.Name
		if diag.Category != "" {
			category += "." + category
		}

		reports[s.URI()] = append(reports[s.URI()], Diagnostic{
			Source:   category,
			Span:     s,
			Message:  fmt.Sprintf(diag.Message),
			Severity: SeverityWarning,
		})
	})

	return reports, nil
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

func runAnalyses(ctx context.Context, v View, pkg Package, report func(a *analysis.Analyzer, diag analysis.Diagnostic)) error {
	// the traditional vet suite:
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

	roots := analyze(ctx, v, []Package{pkg}, analyzers)

	// Report diagnostics and errors from root analyzers.
	for _, r := range roots {
		for _, diag := range r.diagnostics {
			if r.err != nil {
				// TODO(matloob): This isn't quite right: we might return a failed prerequisites error,
				// which isn't super useful...
				return r.err
			}
			report(r.Analyzer, diag)
		}
	}

	return nil
}
