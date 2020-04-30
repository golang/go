// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tests

import (
	"bytes"
	"fmt"
	"go/token"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/diff/myers"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// DiffLinks takes the links we got and checks if they are located within the source or a Note.
// If the link is within a Note, the link is removed.
// Returns an diff comment if there are differences and empty string if no diffs.
func DiffLinks(mapper *protocol.ColumnMapper, wantLinks []Link, gotLinks []protocol.DocumentLink) string {
	var notePositions []token.Position
	links := make(map[span.Span]string, len(wantLinks))
	for _, link := range wantLinks {
		links[link.Src] = link.Target
		notePositions = append(notePositions, link.NotePosition)
	}
	for _, link := range gotLinks {
		spn, err := mapper.RangeSpan(link.Range)
		if err != nil {
			return fmt.Sprintf("%v", err)
		}
		linkInNote := false
		for _, notePosition := range notePositions {
			// Drop the links found inside expectation notes arguments as this links are not collected by expect package.
			if notePosition.Line == spn.Start().Line() &&
				notePosition.Column <= spn.Start().Column() {
				delete(links, spn)
				linkInNote = true
			}
		}
		if linkInNote {
			continue
		}
		if target, ok := links[spn]; ok {
			delete(links, spn)
			if target != link.Target {
				return fmt.Sprintf("for %v want %v, got %v\n", spn, target, link.Target)
			}
		} else {
			return fmt.Sprintf("unexpected link %v:%v\n", spn, link.Target)
		}
	}
	for spn, target := range links {
		return fmt.Sprintf("missing link %v:%v\n", spn, target)
	}
	return ""
}

// DiffSymbols prints the diff between expected and actual symbols test results.
func DiffSymbols(t *testing.T, uri span.URI, want, got []protocol.DocumentSymbol) string {
	sort.Slice(want, func(i, j int) bool { return want[i].Name < want[j].Name })
	sort.Slice(got, func(i, j int) bool { return got[i].Name < got[j].Name })
	if len(got) != len(want) {
		return summarizeSymbols(-1, want, got, "different lengths got %v want %v", len(got), len(want))
	}
	for i, w := range want {
		g := got[i]
		if w.Name != g.Name {
			return summarizeSymbols(i, want, got, "incorrect name got %v want %v", g.Name, w.Name)
		}
		if w.Kind != g.Kind {
			return summarizeSymbols(i, want, got, "incorrect kind got %v want %v", g.Kind, w.Kind)
		}
		if protocol.CompareRange(w.SelectionRange, g.SelectionRange) != 0 {
			return summarizeSymbols(i, want, got, "incorrect span got %v want %v", g.SelectionRange, w.SelectionRange)
		}
		if msg := DiffSymbols(t, uri, w.Children, g.Children); msg != "" {
			return fmt.Sprintf("children of %s: %s", w.Name, msg)
		}
	}
	return ""
}

func summarizeSymbols(i int, want, got []protocol.DocumentSymbol, reason string, args ...interface{}) string {
	msg := &bytes.Buffer{}
	fmt.Fprint(msg, "document symbols failed")
	if i >= 0 {
		fmt.Fprintf(msg, " at %d", i)
	}
	fmt.Fprint(msg, " because of ")
	fmt.Fprintf(msg, reason, args...)
	fmt.Fprint(msg, ":\nexpected:\n")
	for _, s := range want {
		fmt.Fprintf(msg, "  %v %v %v\n", s.Name, s.Kind, s.SelectionRange)
	}
	fmt.Fprintf(msg, "got:\n")
	for _, s := range got {
		fmt.Fprintf(msg, "  %v %v %v\n", s.Name, s.Kind, s.SelectionRange)
	}
	return msg.String()
}

// FilterWorkspaceSymbols filters to got contained in the given dirs.
func FilterWorkspaceSymbols(got []protocol.SymbolInformation, dirs map[string]struct{}) []protocol.SymbolInformation {
	var result []protocol.SymbolInformation
	for _, si := range got {
		if _, ok := dirs[filepath.Dir(si.Location.URI.SpanURI().Filename())]; ok {
			result = append(result, si)
		}
	}
	return result
}

// DiffWorkspaceSymbols prints the diff between expected and actual workspace
// symbols test results.
func DiffWorkspaceSymbols(want, got []protocol.SymbolInformation) string {
	sort.Slice(want, func(i, j int) bool { return fmt.Sprintf("%v", want[i]) < fmt.Sprintf("%v", want[j]) })
	sort.Slice(got, func(i, j int) bool { return fmt.Sprintf("%v", got[i]) < fmt.Sprintf("%v", got[j]) })
	if len(got) != len(want) {
		return summarizeWorkspaceSymbols(-1, want, got, "different lengths got %v want %v", len(got), len(want))
	}
	for i, w := range want {
		g := got[i]
		if w.Name != g.Name {
			return summarizeWorkspaceSymbols(i, want, got, "incorrect name got %v want %v", g.Name, w.Name)
		}
		if w.Kind != g.Kind {
			return summarizeWorkspaceSymbols(i, want, got, "incorrect kind got %v want %v", g.Kind, w.Kind)
		}
		if w.Location.URI != g.Location.URI {
			return summarizeWorkspaceSymbols(i, want, got, "incorrect uri got %v want %v", g.Location.URI, w.Location.URI)
		}
		if protocol.CompareRange(w.Location.Range, g.Location.Range) != 0 {
			return summarizeWorkspaceSymbols(i, want, got, "incorrect range got %v want %v", g.Location.Range, w.Location.Range)
		}
	}
	return ""
}

func summarizeWorkspaceSymbols(i int, want, got []protocol.SymbolInformation, reason string, args ...interface{}) string {
	msg := &bytes.Buffer{}
	fmt.Fprint(msg, "workspace symbols failed")
	if i >= 0 {
		fmt.Fprintf(msg, " at %d", i)
	}
	fmt.Fprint(msg, " because of ")
	fmt.Fprintf(msg, reason, args...)
	fmt.Fprint(msg, ":\nexpected:\n")
	for _, s := range want {
		fmt.Fprintf(msg, "  %v %v %v:%v\n", s.Name, s.Kind, s.Location.URI, s.Location.Range)
	}
	fmt.Fprintf(msg, "got:\n")
	for _, s := range got {
		fmt.Fprintf(msg, "  %v %v %v:%v\n", s.Name, s.Kind, s.Location.URI, s.Location.Range)
	}
	return msg.String()
}

// DiffDiagnostics prints the diff between expected and actual diagnostics test
// results.
func DiffDiagnostics(uri span.URI, want, got []*source.Diagnostic) string {
	source.SortDiagnostics(want)
	source.SortDiagnostics(got)

	if len(got) != len(want) {
		return summarizeDiagnostics(-1, uri, want, got, "different lengths got %v want %v", len(got), len(want))
	}
	for i, w := range want {
		g := got[i]
		if w.Message != g.Message {
			return summarizeDiagnostics(i, uri, want, got, "incorrect Message got %v want %v", g.Message, w.Message)
		}
		if w.Severity != g.Severity {
			return summarizeDiagnostics(i, uri, want, got, "incorrect Severity got %v want %v", g.Severity, w.Severity)
		}
		if w.Source != g.Source {
			return summarizeDiagnostics(i, uri, want, got, "incorrect Source got %v want %v", g.Source, w.Source)
		}
		// Don't check the range on the badimport test.
		if strings.Contains(uri.Filename(), "badimport") {
			continue
		}
		if protocol.ComparePosition(w.Range.Start, g.Range.Start) != 0 {
			return summarizeDiagnostics(i, uri, want, got, "incorrect Start got %v want %v", g.Range.Start, w.Range.Start)
		}
		if !protocol.IsPoint(g.Range) { // Accept any 'want' range if the diagnostic returns a zero-length range.
			if protocol.ComparePosition(w.Range.End, g.Range.End) != 0 {
				return summarizeDiagnostics(i, uri, want, got, "incorrect End got %v want %v", g.Range.End, w.Range.End)
			}
		}
	}
	return ""
}

func summarizeDiagnostics(i int, uri span.URI, want, got []*source.Diagnostic, reason string, args ...interface{}) string {
	msg := &bytes.Buffer{}
	fmt.Fprint(msg, "diagnostics failed")
	if i >= 0 {
		fmt.Fprintf(msg, " at %d", i)
	}
	fmt.Fprint(msg, " because of ")
	fmt.Fprintf(msg, reason, args...)
	fmt.Fprint(msg, ":\nexpected:\n")
	for _, d := range want {
		fmt.Fprintf(msg, "  %s:%v: %s\n", uri, d.Range, d.Message)
	}
	fmt.Fprintf(msg, "got:\n")
	for _, d := range got {
		fmt.Fprintf(msg, "  %s:%v: %s\n", uri, d.Range, d.Message)
	}
	return msg.String()
}

func DiffCodeLens(uri span.URI, want, got []protocol.CodeLens) string {
	sortCodeLens(want)
	sortCodeLens(got)

	if len(got) != len(want) {
		return summarizeCodeLens(-1, uri, want, got, "different lengths got %v want %v", len(got), len(want))
	}
	for i, w := range want {
		g := got[i]
		if w.Command.Title != g.Command.Title {
			return summarizeCodeLens(i, uri, want, got, "incorrect Command Title got %v want %v", g.Command.Title, w.Command.Title)
		}
		if w.Command.Command != g.Command.Command {
			return summarizeCodeLens(i, uri, want, got, "incorrect Command Title got %v want %v", g.Command.Command, w.Command.Command)
		}
		if protocol.ComparePosition(w.Range.Start, g.Range.Start) != 0 {
			return summarizeCodeLens(i, uri, want, got, "incorrect Start got %v want %v", g.Range.Start, w.Range.Start)
		}
		if !protocol.IsPoint(g.Range) { // Accept any 'want' range if the codelens returns a zero-length range.
			if protocol.ComparePosition(w.Range.End, g.Range.End) != 0 {
				return summarizeCodeLens(i, uri, want, got, "incorrect End got %v want %v", g.Range.End, w.Range.End)
			}
		}
	}
	return ""
}

func sortCodeLens(c []protocol.CodeLens) {
	sort.Slice(c, func(i int, j int) bool {
		if r := protocol.CompareRange(c[i].Range, c[j].Range); r != 0 {
			return r < 0
		}
		if c[i].Command.Command < c[j].Command.Command {
			return true
		}
		if c[i].Command.Command == c[j].Command.Command {
			return true
		}
		return c[i].Command.Title <= c[j].Command.Title
	})
}

func summarizeCodeLens(i int, uri span.URI, want, got []protocol.CodeLens, reason string, args ...interface{}) string {
	msg := &bytes.Buffer{}
	fmt.Fprint(msg, "codelens failed")
	if i >= 0 {
		fmt.Fprintf(msg, " at %d", i)
	}
	fmt.Fprint(msg, " because of ")
	fmt.Fprintf(msg, reason, args...)
	fmt.Fprint(msg, ":\nexpected:\n")
	for _, d := range want {
		fmt.Fprintf(msg, "  %s:%v: %s | %s\n", uri, d.Range, d.Command.Command, d.Command.Title)
	}
	fmt.Fprintf(msg, "got:\n")
	for _, d := range got {
		fmt.Fprintf(msg, "  %s:%v: %s | %s\n", uri, d.Range, d.Command.Command, d.Command.Title)
	}
	return msg.String()
}

func DiffSignatures(spn span.Span, want, got *protocol.SignatureHelp) string {
	decorate := func(f string, args ...interface{}) string {
		return fmt.Sprintf("invalid signature at %s: %s", spn, fmt.Sprintf(f, args...))
	}
	if len(got.Signatures) != 1 {
		return decorate("wanted 1 signature, got %d", len(got.Signatures))
	}
	if got.ActiveSignature != 0 {
		return decorate("wanted active signature of 0, got %d", int(got.ActiveSignature))
	}
	if want.ActiveParameter != got.ActiveParameter {
		return decorate("wanted active parameter of %d, got %d", want.ActiveParameter, int(got.ActiveParameter))
	}
	g := got.Signatures[0]
	w := want.Signatures[0]
	if w.Label != g.Label {
		wLabel := w.Label + "\n"
		d := myers.ComputeEdits("", wLabel, g.Label+"\n")
		return decorate("mismatched labels:\n%q", diff.ToUnified("want", "got", wLabel, d))
	}
	var paramParts []string
	for _, p := range g.Parameters {
		paramParts = append(paramParts, p.Label)
	}
	paramsStr := strings.Join(paramParts, ", ")
	if !strings.Contains(g.Label, paramsStr) {
		return decorate("expected signature %q to contain params %q", g.Label, paramsStr)
	}
	return ""
}

func ToProtocolCompletionItems(items []source.CompletionItem) []protocol.CompletionItem {
	var result []protocol.CompletionItem
	for _, item := range items {
		result = append(result, ToProtocolCompletionItem(item))
	}
	return result
}

func ToProtocolCompletionItem(item source.CompletionItem) protocol.CompletionItem {
	pItem := protocol.CompletionItem{
		Label:         item.Label,
		Kind:          item.Kind,
		Detail:        item.Detail,
		Documentation: item.Documentation,
		InsertText:    item.InsertText,
		TextEdit: &protocol.TextEdit{
			NewText: item.Snippet(),
		},
		// Negate score so best score has lowest sort text like real API.
		SortText: fmt.Sprint(-item.Score),
	}
	if pItem.InsertText == "" {
		pItem.InsertText = pItem.Label
	}
	return pItem
}

func FilterBuiltins(src span.Span, items []protocol.CompletionItem) []protocol.CompletionItem {
	var (
		got          []protocol.CompletionItem
		wantBuiltins = strings.Contains(string(src.URI()), "builtins")
		wantKeywords = strings.Contains(string(src.URI()), "keywords")
	)
	for _, item := range items {
		if !wantBuiltins && isBuiltin(item.Label, item.Detail, item.Kind) {
			continue
		}

		if !wantKeywords && token.Lookup(item.Label).IsKeyword() {
			continue
		}

		got = append(got, item)
	}
	return got
}

func isBuiltin(label, detail string, kind protocol.CompletionItemKind) bool {
	if detail == "" && kind == protocol.ClassCompletion {
		return true
	}
	// Remaining builtin constants, variables, interfaces, and functions.
	trimmed := label
	if i := strings.Index(trimmed, "("); i >= 0 {
		trimmed = trimmed[:i]
	}
	switch trimmed {
	case "append", "cap", "close", "complex", "copy", "delete",
		"error", "false", "imag", "iota", "len", "make", "new",
		"nil", "panic", "print", "println", "real", "recover", "true":
		return true
	}
	return false
}

func CheckCompletionOrder(want, got []protocol.CompletionItem, strictScores bool) string {
	var (
		matchedIdxs []int
		lastGotIdx  int
		lastGotSort float64
		inOrder     = true
		errorMsg    = "completions out of order"
	)
	for _, w := range want {
		var found bool
		for i, g := range got {
			if w.Label == g.Label && w.Detail == g.Detail && w.Kind == g.Kind {
				matchedIdxs = append(matchedIdxs, i)
				found = true

				if i < lastGotIdx {
					inOrder = false
				}
				lastGotIdx = i

				sort, _ := strconv.ParseFloat(g.SortText, 64)
				if strictScores && len(matchedIdxs) > 1 && sort <= lastGotSort {
					inOrder = false
					errorMsg = "candidate scores not strictly decreasing"
				}
				lastGotSort = sort

				break
			}
		}
		if !found {
			return summarizeCompletionItems(-1, []protocol.CompletionItem{w}, got, "didn't find expected completion")
		}
	}

	sort.Ints(matchedIdxs)
	matched := make([]protocol.CompletionItem, 0, len(matchedIdxs))
	for _, idx := range matchedIdxs {
		matched = append(matched, got[idx])
	}

	if !inOrder {
		return summarizeCompletionItems(-1, want, matched, errorMsg)
	}

	return ""
}

func DiffSnippets(want string, got *protocol.CompletionItem) string {
	if want == "" {
		if got != nil {
			x := got.TextEdit
			return fmt.Sprintf("expected no snippet but got %s", x.NewText)
		}
	} else {
		if got == nil {
			return fmt.Sprintf("couldn't find completion matching %q", want)
		}
		x := got.TextEdit
		if want != x.NewText {
			return fmt.Sprintf("expected snippet %q, got %q", want, x.NewText)
		}
	}
	return ""
}

func FindItem(list []protocol.CompletionItem, want source.CompletionItem) *protocol.CompletionItem {
	for _, item := range list {
		if item.Label == want.Label {
			return &item
		}
	}
	return nil
}

// DiffCompletionItems prints the diff between expected and actual completion
// test results.
func DiffCompletionItems(want, got []protocol.CompletionItem) string {
	if len(got) != len(want) {
		return summarizeCompletionItems(-1, want, got, "different lengths got %v want %v", len(got), len(want))
	}
	for i, w := range want {
		g := got[i]
		if w.Label != g.Label {
			return summarizeCompletionItems(i, want, got, "incorrect Label got %v want %v", g.Label, w.Label)
		}
		if w.Detail != g.Detail {
			return summarizeCompletionItems(i, want, got, "incorrect Detail got %v want %v", g.Detail, w.Detail)
		}
		if w.Documentation != "" && !strings.HasPrefix(w.Documentation, "@") {
			if w.Documentation != g.Documentation {
				return summarizeCompletionItems(i, want, got, "incorrect Documentation got %v want %v", g.Documentation, w.Documentation)
			}
		}
		if w.Kind != g.Kind {
			return summarizeCompletionItems(i, want, got, "incorrect Kind got %v want %v", g.Kind, w.Kind)
		}
	}
	return ""
}

func summarizeCompletionItems(i int, want, got []protocol.CompletionItem, reason string, args ...interface{}) string {
	msg := &bytes.Buffer{}
	fmt.Fprint(msg, "completion failed")
	if i >= 0 {
		fmt.Fprintf(msg, " at %d", i)
	}
	fmt.Fprint(msg, " because of ")
	fmt.Fprintf(msg, reason, args...)
	fmt.Fprint(msg, ":\nexpected:\n")
	for _, d := range want {
		fmt.Fprintf(msg, "  %v\n", d)
	}
	fmt.Fprintf(msg, "got:\n")
	for _, d := range got {
		fmt.Fprintf(msg, "  %v\n", d)
	}
	return msg.String()
}

func FormatFolderName(folder string) string {
	if index := strings.Index(folder, "testdata"); index != -1 {
		return folder[index:]
	}
	return folder
}

func EnableAllAnalyzers(snapshot source.Snapshot, opts *source.Options) {
	if opts.UserEnabledAnalyses == nil {
		opts.UserEnabledAnalyses = make(map[string]bool)
	}
	for _, a := range opts.DefaultAnalyzers {
		if !a.Enabled(snapshot) {
			opts.UserEnabledAnalyses[a.Analyzer.Name] = true
		}
	}
	for _, a := range opts.TypeErrorAnalyzers {
		if !a.Enabled(snapshot) {
			opts.UserEnabledAnalyses[a.Analyzer.Name] = true
		}
	}
}

func Diff(want, got string) string {
	if want == got {
		return ""
	}
	// Add newlines to avoid newline messages in diff.
	want += "\n"
	got += "\n"
	d := myers.ComputeEdits("", want, got)
	return fmt.Sprintf("%q", diff.ToUnified("want", "got", want, d))
}
