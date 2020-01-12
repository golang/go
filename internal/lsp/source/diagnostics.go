// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"strings"

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

func Diagnostics(ctx context.Context, snapshot Snapshot, ph PackageHandle, withAnalysis bool) (map[FileIdentity][]Diagnostic, string, error) {
	// If we are missing dependencies, it may because the user's workspace is
	// not correctly configured. Report errors, if possible.
	var (
		warn       bool
		warningMsg string
	)
	if len(ph.MissingDependencies()) > 0 {
		warn = true
	}
	pkg, err := ph.Check(ctx)
	if err != nil {
		return nil, "", err
	}
	// If we have a package with a single file and errors about "undeclared" symbols,
	// we may have an ad-hoc package with multiple files. Show a warning message.
	// TODO(golang/go#36416): Remove this when golang.org/cl/202277 is merged.
	if len(pkg.CompiledGoFiles()) == 1 && hasUndeclaredErrors(pkg) {
		warn = true
	}
	if warn {
		warningMsg, err = checkCommonErrors(ctx, snapshot.View())
		if err != nil {
			return nil, "", err
		}
	}
	// Prepare the reports we will send for the files in this package.
	reports := make(map[FileIdentity][]Diagnostic)
	for _, fh := range pkg.CompiledGoFiles() {
		clearReports(snapshot, reports, fh.File().Identity())
	}
	// Prepare any additional reports for the errors in this package.
	for _, e := range pkg.GetErrors() {
		// We only need to handle lower-level errors.
		if e.Kind != ListError {
			continue
		}
		// If no file is associated with the error, pick an open file from the package.
		if e.URI.Filename() == "" {
			for _, ph := range pkg.CompiledGoFiles() {
				if snapshot.View().Session().IsOpen(ph.File().Identity().URI) {
					e.URI = ph.File().Identity().URI
				}
			}
		}
		fh, err := snapshot.GetFile(e.URI)
		if err != nil {
			return nil, warningMsg, err
		}
		clearReports(snapshot, reports, fh.Identity())
	}
	// Run diagnostics for the package that this URI belongs to.
	hadDiagnostics, err := diagnostics(ctx, snapshot, reports, pkg)
	if err != nil {
		return nil, warningMsg, err
	}
	if !hadDiagnostics && withAnalysis {
		// If we don't have any list, parse, or type errors, run analyses.
		if err := analyses(ctx, snapshot, reports, ph, snapshot.View().Options().DisabledAnalyses); err != nil {
			// Exit early if the context has been canceled.
			if ctx.Err() != nil {
				return nil, warningMsg, ctx.Err()
			}
			log.Error(ctx, "failed to run analyses", err, telemetry.Package.Of(ph.ID()))
		}
	}
	return reports, warningMsg, nil
}

func FileDiagnostics(ctx context.Context, snapshot Snapshot, uri span.URI) (FileIdentity, []Diagnostic, error) {
	fh, err := snapshot.GetFile(uri)
	if err != nil {
		return FileIdentity{}, nil, err
	}
	phs, err := snapshot.PackageHandles(ctx, fh)
	if err != nil {
		return FileIdentity{}, nil, err
	}
	ph, err := NarrowestPackageHandle(phs)
	if err != nil {
		return FileIdentity{}, nil, err
	}
	reports, _, err := Diagnostics(ctx, snapshot, ph, true)
	if err != nil {
		return FileIdentity{}, nil, err
	}
	diagnostics, ok := reports[fh.Identity()]
	if !ok {
		return FileIdentity{}, nil, errors.Errorf("no diagnostics for %s", uri)
	}
	return fh.Identity(), diagnostics, nil
}

type diagnosticSet struct {
	listErrors, parseErrors, typeErrors []*Diagnostic
}

func diagnostics(ctx context.Context, snapshot Snapshot, reports map[FileIdentity][]Diagnostic, pkg Package) (bool, error) {
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
		fh, err := snapshot.GetFile(e.URI)
		if err != nil {
			return false, err
		}
		set, ok := diagSets[fh.Identity()]
		if !ok {
			set = &diagnosticSet{}
			diagSets[fh.Identity()] = set
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
		addReports(ctx, snapshot, reports, fileID, diags...)
	}
	return nonEmptyDiagnostics, nil
}

func analyses(ctx context.Context, snapshot Snapshot, reports map[FileIdentity][]Diagnostic, ph PackageHandle, disabledAnalyses map[string]struct{}) error {
	var analyzers []*analysis.Analyzer
	for _, a := range snapshot.View().Options().Analyzers {
		if _, ok := disabledAnalyses[a.Name]; ok {
			continue
		}
		analyzers = append(analyzers, a)
	}

	diagnostics, err := snapshot.Analyze(ctx, ph.ID(), analyzers)
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
		fh, err := snapshot.GetFile(e.URI)
		if err != nil {
			return err
		}
		addReports(ctx, snapshot, reports, fh.Identity(), &Diagnostic{
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

func addReports(ctx context.Context, snapshot Snapshot, reports map[FileIdentity][]Diagnostic, fileID FileIdentity, diagnostics ...*Diagnostic) error {
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
