// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

type Diagnostic struct {
	Range    protocol.Range
	Message  string
	Source   string
	Severity protocol.DiagnosticSeverity
	Tags     []protocol.DiagnosticTag

	SuggestedFixes []SuggestedFix
	Related        []RelatedInformation
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

func Diagnostics(ctx context.Context, snapshot Snapshot, f File, disabledAnalyses map[string]struct{}) (map[FileIdentity][]Diagnostic, string, error) {
	ctx, done := trace.StartSpan(ctx, "source.Diagnostics", telemetry.File.Of(f.URI()))
	defer done()

	fh := snapshot.Handle(ctx, f)
	cphs, err := snapshot.PackageHandles(ctx, f)
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
		warningMsg, err = checkCommonErrors(ctx, snapshot.View(), f.URI())
		if err != nil {
			log.Error(ctx, "error checking common errors", err, telemetry.File.Of(f.URI))
		}
	}
	pkg, err := cph.Check(ctx)
	if err != nil {
		log.Error(ctx, "no package for file", err)
		return singleDiagnostic(fh.Identity(), "%s is not part of a package", f.URI()), "", nil
	}
	// Prepare the reports we will send for the files in this package.
	reports := make(map[FileIdentity][]Diagnostic)
	for _, fh := range pkg.CompiledGoFiles() {
		clearReports(snapshot, reports, fh.File().Identity())
	}
	// Prepare any additional reports for the errors in this package.
	for _, e := range pkg.GetErrors() {
		if e.Kind != ListError {
			continue
		}
		clearReports(snapshot, reports, e.File)
	}
	// Run diagnostics for the package that this URI belongs to.
	if !diagnostics(ctx, snapshot, pkg, reports) {
		// If we don't have any list, parse, or type errors, run analyses.
		if err := analyses(ctx, snapshot, cph, disabledAnalyses, reports); err != nil {
			log.Error(ctx, "failed to run analyses", err, telemetry.File.Of(f.URI()))
		}
	}
	// Updates to the diagnostics for this package may need to be propagated.
	for _, id := range snapshot.GetReverseDependencies(pkg.ID()) {
		cph, err := snapshot.PackageHandle(ctx, id)
		if err != nil {
			return nil, warningMsg, err
		}
		pkg, err := cph.Check(ctx)
		if err != nil {
			return nil, warningMsg, err
		}
		for _, fh := range pkg.CompiledGoFiles() {
			clearReports(snapshot, reports, fh.File().Identity())
		}
		diagnostics(ctx, snapshot, pkg, reports)
	}
	return reports, warningMsg, nil
}

type diagnosticSet struct {
	listErrors, parseErrors, typeErrors []*Diagnostic
}

func diagnostics(ctx context.Context, snapshot Snapshot, pkg Package, reports map[FileIdentity][]Diagnostic) bool {
	ctx, done := trace.StartSpan(ctx, "source.diagnostics", telemetry.Package.Of(pkg.ID()))
	_ = ctx // circumvent SA4006
	defer done()

	diagSets := make(map[FileIdentity]*diagnosticSet)
	for _, e := range pkg.GetErrors() {
		diag := &Diagnostic{
			Message:  e.Message,
			Range:    e.Range,
			Severity: protocol.SeverityError,
		}
		set, ok := diagSets[e.File]
		if !ok {
			set = &diagnosticSet{}
			diagSets[e.File] = set
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
	var nonEmptyDiagnostics bool // track if we actually send non-empty diagnostics
	for fileID, set := range diagSets {
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
		addReports(ctx, reports, snapshot, fileID, diags...)
	}
	return nonEmptyDiagnostics
}

func analyses(ctx context.Context, snapshot Snapshot, cph CheckPackageHandle, disabledAnalyses map[string]struct{}, reports map[FileIdentity][]Diagnostic) error {
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
	for _, e := range diagnostics {
		// This is a bit of a hack, but clients > 3.15 will be able to grey out unnecessary code.
		// If we are deleting code as part of all of our suggested fixes, assume that this is dead code.
		// TODO(golang/go/#34508): Return these codes from the diagnostics themselves.
		var tags []protocol.DiagnosticTag
		if onlyDeletions(e.SuggestedFixes) {
			tags = append(tags, protocol.Unnecessary)
		}
		addReports(ctx, reports, snapshot, e.File, &Diagnostic{
			Range:          e.Range,
			Message:        e.Message,
			Source:         e.Category,
			Severity:       protocol.SeverityWarning,
			Tags:           tags,
			SuggestedFixes: e.SuggestedFixes,
			Related:        e.Related,
		})
	}
	return nil
}

func clearReports(snapshot Snapshot, reports map[FileIdentity][]Diagnostic, fileID FileIdentity) {
	if snapshot.View().Ignore(fileID.URI) {
		return
	}
	reports[fileID] = []Diagnostic{}
}

func addReports(ctx context.Context, reports map[FileIdentity][]Diagnostic, snapshot Snapshot, fileID FileIdentity, diagnostics ...*Diagnostic) error {
	if snapshot.View().Ignore(fileID.URI) {
		return nil
	}
	if _, ok := reports[fileID]; !ok {
		return errors.Errorf("diagnostics for unexpected file %s", fileID.URI)
	}
	for _, diag := range diagnostics {
		if diag == nil {
			continue
		}
		reports[fileID] = append(reports[fileID], *diag)
	}
	return nil
}

func singleDiagnostic(fileID FileIdentity, format string, a ...interface{}) map[FileIdentity][]Diagnostic {
	return map[FileIdentity][]Diagnostic{
		fileID: []Diagnostic{
			{
				Source:   "gopls",
				Range:    protocol.Range{},
				Message:  fmt.Sprintf(format, a...),
				Severity: protocol.SeverityError,
			},
		},
	}
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
	return true
}
