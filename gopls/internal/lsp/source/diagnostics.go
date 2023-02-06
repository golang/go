// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
)

type SuggestedFix struct {
	Title      string
	Edits      map[span.URI][]protocol.TextEdit
	Command    *protocol.Command
	ActionKind protocol.CodeActionKind
}

type RelatedInformation struct {
	// TODO(adonovan): replace these two fields by a protocol.Location.
	URI     span.URI
	Range   protocol.Range
	Message string
}

// Analyze reports go/analysis-framework diagnostics in the specified package.
func Analyze(ctx context.Context, snapshot Snapshot, pkgid PackageID, includeConvenience bool) (map[span.URI][]*Diagnostic, error) {
	// Exit early if the context has been canceled. This also protects us
	// from a race on Options, see golang/go#36699.
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	options := snapshot.View().Options()
	categories := []map[string]*Analyzer{
		options.DefaultAnalyzers,
		options.StaticcheckAnalyzers,
		options.TypeErrorAnalyzers,
	}
	if includeConvenience { // e.g. for codeAction
		categories = append(categories, options.ConvenienceAnalyzers) // e.g. fillstruct
	}

	var analyzers []*Analyzer
	for _, cat := range categories {
		for _, a := range cat {
			analyzers = append(analyzers, a)
		}
	}

	analysisDiagnostics, err := snapshot.Analyze(ctx, pkgid, analyzers)
	if err != nil {
		return nil, err
	}

	// Report diagnostics and errors from root analyzers.
	reports := make(map[span.URI][]*Diagnostic)
	for _, diag := range analysisDiagnostics {
		reports[diag.URI] = append(reports[diag.URI], diag)
	}
	return reports, nil
}

// FileDiagnostics reports diagnostics in the specified file,
// as used by the "gopls check" command.
//
// TODO(adonovan): factor in common with (*Server).codeAction, which
// executes { PackageForFile; Analyze } too?
//
// TODO(adonovan): opt: this function is called in a loop from the
// "gopls/diagnoseFiles" nonstandard request handler. It would be more
// efficient to compute the set of packages and TypeCheck and
// Analyze them all at once.
func FileDiagnostics(ctx context.Context, snapshot Snapshot, uri span.URI) (FileHandle, []*Diagnostic, error) {
	fh, err := snapshot.GetFile(ctx, uri)
	if err != nil {
		return nil, nil, err
	}
	pkg, _, err := PackageForFile(ctx, snapshot, uri, TypecheckFull, NarrowestPackage)
	if err != nil {
		return nil, nil, err
	}
	pkgDiags, err := pkg.DiagnosticsForFile(ctx, snapshot, uri)
	if err != nil {
		return nil, nil, err
	}
	adiags, err := Analyze(ctx, snapshot, pkg.Metadata().ID, false)
	if err != nil {
		return nil, nil, err
	}
	var fileDiags []*Diagnostic // combine load/parse/type + analysis diagnostics
	CombineDiagnostics(pkgDiags, adiags[uri], &fileDiags, &fileDiags)
	return fh, fileDiags, nil
}

// CombineDiagnostics combines and filters list/parse/type diagnostics from
// tdiags with adiags, and appends the two lists to *outT and *outA,
// respectively.
//
// Type-error analyzers produce diagnostics that are redundant
// with type checker diagnostics, but more detailed (e.g. fixes).
// Rather than report two diagnostics for the same problem,
// we combine them by augmenting the type-checker diagnostic
// and discarding the analyzer diagnostic.
//
// If an analysis diagnostic has the same range and message as
// a list/parse/type diagnostic, the suggested fix information
// (et al) of the latter is merged into a copy of the former.
// This handles the case where a type-error analyzer suggests
// a fix to a type error, and avoids duplication.
//
// The use of out-slices, though irregular, allows the caller to
// easily choose whether to keep the results separate or combined.
//
// The arguments are not modified.
func CombineDiagnostics(tdiags []*Diagnostic, adiags []*Diagnostic, outT, outA *[]*Diagnostic) {

	// Build index of (list+parse+)type errors.
	type key struct {
		Range   protocol.Range
		message string
	}
	index := make(map[key]int) // maps (Range,Message) to index in tdiags slice
	for i, diag := range tdiags {
		index[key{diag.Range, diag.Message}] = i
	}

	// Filter out analysis diagnostics that match type errors,
	// retaining their suggested fix (etc) fields.
	for _, diag := range adiags {
		if i, ok := index[key{diag.Range, diag.Message}]; ok {
			copy := *tdiags[i]
			copy.SuggestedFixes = diag.SuggestedFixes
			copy.Tags = diag.Tags
			tdiags[i] = &copy
			continue
		}

		*outA = append(*outA, diag)
	}

	*outT = append(*outT, tdiags...)
}
