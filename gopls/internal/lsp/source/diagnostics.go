// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"encoding/json"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/progress"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
)

type SuggestedFix struct {
	Title      string
	Edits      map[span.URI][]protocol.TextEdit
	Command    *protocol.Command
	ActionKind protocol.CodeActionKind
}

// Analyze reports go/analysis-framework diagnostics in the specified package.
//
// If the provided tracker is non-nil, it may be used to provide notifications
// of the ongoing analysis pass.
func Analyze(ctx context.Context, snapshot Snapshot, pkgIDs map[PackageID]unit, tracker *progress.Tracker) (map[span.URI][]*Diagnostic, error) {
	// Exit early if the context has been canceled. This also protects us
	// from a race on Options, see golang/go#36699.
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	options := snapshot.Options()
	categories := []map[string]*Analyzer{
		options.DefaultAnalyzers,
		options.StaticcheckAnalyzers,
		options.TypeErrorAnalyzers,
	}

	var analyzers []*Analyzer
	for _, cat := range categories {
		for _, a := range cat {
			analyzers = append(analyzers, a)
		}
	}

	analysisDiagnostics, err := snapshot.Analyze(ctx, pkgIDs, analyzers, tracker)
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

// quickFixesJSON is a JSON-serializable list of quick fixes
// to be saved in the protocol.Diagnostic.Data field.
type quickFixesJSON struct {
	// TODO(rfindley): pack some sort of identifier here for later
	// lookup/validation?
	Fixes []protocol.CodeAction
}

// BundleQuickFixes attempts to bundle sd.SuggestedFixes into the
// sd.BundledFixes field, so that it can be round-tripped through the client.
// It returns false if the quick-fixes cannot be bundled.
func BundleQuickFixes(sd *Diagnostic) bool {
	if len(sd.SuggestedFixes) == 0 {
		return true
	}
	var actions []protocol.CodeAction
	for _, fix := range sd.SuggestedFixes {
		if fix.Edits != nil {
			// For now, we only support bundled code actions that execute commands.
			//
			// In order to cleanly support bundled edits, we'd have to guarantee that
			// the edits were generated on the current snapshot. But this naively
			// implies that every fix would have to include a snapshot ID, which
			// would require us to republish all diagnostics on each new snapshot.
			//
			// TODO(rfindley): in order to avoid this additional chatter, we'd need
			// to build some sort of registry or other mechanism on the snapshot to
			// check whether a diagnostic is still valid.
			return false
		}
		action := protocol.CodeAction{
			Title:   fix.Title,
			Kind:    fix.ActionKind,
			Command: fix.Command,
		}
		actions = append(actions, action)
	}
	fixes := quickFixesJSON{
		Fixes: actions,
	}
	data, err := json.Marshal(fixes)
	if err != nil {
		bug.Reportf("marshalling quick fixes: %v", err)
		return false
	}
	msg := json.RawMessage(data)
	sd.BundledFixes = &msg
	return true
}

// BundledQuickFixes extracts any bundled codeActions from the
// diag.Data field.
func BundledQuickFixes(diag protocol.Diagnostic) []protocol.CodeAction {
	if diag.Data == nil {
		return nil
	}
	var fix quickFixesJSON
	if err := json.Unmarshal(*diag.Data, &fix); err != nil {
		bug.Reportf("unmarshalling quick fix: %v", err)
		return nil
	}

	var actions []protocol.CodeAction
	for _, action := range fix.Fixes {
		// See BundleQuickFixes: for now we only support bundling commands.
		if action.Edit != nil {
			bug.Reportf("bundled fix %q includes workspace edits", action.Title)
			continue
		}
		// associate the action with the incoming diagnostic
		// (Note that this does not mutate the fix.Fixes slice).
		action.Diagnostics = []protocol.Diagnostic{diag}
		actions = append(actions, action)
	}

	return actions
}
