// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tests

import (
	"bytes"
	"context"
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
	"golang.org/x/tools/internal/lsp/source/completion"
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
		if !rangeOverlaps(g.Range, w.Range) {
			return summarizeDiagnostics(i, uri, want, got, "range %v does not overlap %v", g.Range, w.Range)
		}
	}
	return ""
}

// rangeOverlaps reports whether r1 and r2 overlap.
func rangeOverlaps(r1, r2 protocol.Range) bool {
	if inRange(r2.Start, r1) || inRange(r1.Start, r2) {
		return true
	}
	return false
}

// inRange reports whether p is contained within [r.Start, r.End), or if p ==
// r.Start == r.End (special handling for the case where the range is a single
// point).
func inRange(p protocol.Position, r protocol.Range) bool {
	if protocol.IsPoint(r) {
		return protocol.ComparePosition(r.Start, p) == 0
	}
	if protocol.ComparePosition(r.Start, p) <= 0 && protocol.ComparePosition(p, r.End) < 0 {
		return true
	}
	return false
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
		if w.Command.Command != g.Command.Command {
			return summarizeCodeLens(i, uri, want, got, "incorrect Command Name got %v want %v", g.Command.Command, w.Command.Command)
		}
		if w.Command.Title != g.Command.Title {
			return summarizeCodeLens(i, uri, want, got, "incorrect Command Title got %v want %v", g.Command.Title, w.Command.Title)
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
		} else if c[i].Command.Command == c[j].Command.Command {
			return c[i].Command.Title < c[j].Command.Title
		} else {
			return false
		}
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

func DiffSignatures(spn span.Span, want, got *protocol.SignatureHelp) (string, error) {
	decorate := func(f string, args ...interface{}) string {
		return fmt.Sprintf("invalid signature at %s: %s", spn, fmt.Sprintf(f, args...))
	}
	if len(got.Signatures) != 1 {
		return decorate("wanted 1 signature, got %d", len(got.Signatures)), nil
	}
	if got.ActiveSignature != 0 {
		return decorate("wanted active signature of 0, got %d", int(got.ActiveSignature)), nil
	}
	if want.ActiveParameter != got.ActiveParameter {
		return decorate("wanted active parameter of %d, got %d", want.ActiveParameter, int(got.ActiveParameter)), nil
	}
	g := got.Signatures[0]
	w := want.Signatures[0]
	if NormalizeAny(w.Label) != NormalizeAny(g.Label) {
		wLabel := w.Label + "\n"
		d, err := myers.ComputeEdits("", wLabel, g.Label+"\n")
		if err != nil {
			return "", err
		}
		return decorate("mismatched labels:\n%q", diff.ToUnified("want", "got", wLabel, d)), err
	}
	var paramParts []string
	for _, p := range g.Parameters {
		paramParts = append(paramParts, p.Label)
	}
	paramsStr := strings.Join(paramParts, ", ")
	if !strings.Contains(g.Label, paramsStr) {
		return decorate("expected signature %q to contain params %q", g.Label, paramsStr), nil
	}
	return "", nil
}

// NormalizeAny replaces occurrences of interface{} in input with any.
//
// In Go 1.18, standard library functions were changed to use the 'any'
// alias in place of interface{}, which affects their type string.
func NormalizeAny(input string) string {
	return strings.ReplaceAll(input, "interface{}", "any")
}

// DiffCallHierarchyItems returns the diff between expected and actual call locations for incoming/outgoing call hierarchies
func DiffCallHierarchyItems(gotCalls []protocol.CallHierarchyItem, expectedCalls []protocol.CallHierarchyItem) string {
	expected := make(map[protocol.Location]bool)
	for _, call := range expectedCalls {
		expected[protocol.Location{URI: call.URI, Range: call.Range}] = true
	}

	got := make(map[protocol.Location]bool)
	for _, call := range gotCalls {
		got[protocol.Location{URI: call.URI, Range: call.Range}] = true
	}
	if len(got) != len(expected) {
		return fmt.Sprintf("expected %d calls but got %d", len(expected), len(got))
	}
	for spn := range got {
		if !expected[spn] {
			return fmt.Sprintf("incorrect calls, expected locations %v but got locations %v", expected, got)
		}
	}
	return ""
}

func ToProtocolCompletionItems(items []completion.CompletionItem) []protocol.CompletionItem {
	var result []protocol.CompletionItem
	for _, item := range items {
		result = append(result, ToProtocolCompletionItem(item))
	}
	return result
}

func ToProtocolCompletionItem(item completion.CompletionItem) protocol.CompletionItem {
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
			if w.Label == g.Label && NormalizeAny(w.Detail) == NormalizeAny(g.Detail) && w.Kind == g.Kind {
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

func FindItem(list []protocol.CompletionItem, want completion.CompletionItem) *protocol.CompletionItem {
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
		if NormalizeAny(w.Detail) != NormalizeAny(g.Detail) {
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

func EnableAllAnalyzers(view source.View, opts *source.Options) {
	if opts.Analyses == nil {
		opts.Analyses = make(map[string]bool)
	}
	for _, a := range opts.DefaultAnalyzers {
		if !a.IsEnabled(view) {
			opts.Analyses[a.Analyzer.Name] = true
		}
	}
	for _, a := range opts.TypeErrorAnalyzers {
		if !a.IsEnabled(view) {
			opts.Analyses[a.Analyzer.Name] = true
		}
	}
	for _, a := range opts.ConvenienceAnalyzers {
		if !a.IsEnabled(view) {
			opts.Analyses[a.Analyzer.Name] = true
		}
	}
	for _, a := range opts.StaticcheckAnalyzers {
		if !a.IsEnabled(view) {
			opts.Analyses[a.Analyzer.Name] = true
		}
	}
}

func WorkspaceSymbolsString(ctx context.Context, data *Data, queryURI span.URI, symbols []protocol.SymbolInformation) (string, error) {
	queryDir := filepath.Dir(queryURI.Filename())
	var filtered []string
	for _, s := range symbols {
		uri := s.Location.URI.SpanURI()
		dir := filepath.Dir(uri.Filename())
		if !source.InDir(queryDir, dir) { // assume queries always issue from higher directories
			continue
		}
		m, err := data.Mapper(uri)
		if err != nil {
			return "", err
		}
		spn, err := m.Span(s.Location)
		if err != nil {
			return "", err
		}
		filtered = append(filtered, fmt.Sprintf("%s %s %s", spn, s.Name, s.Kind))
	}
	sort.Strings(filtered)
	return strings.Join(filtered, "\n") + "\n", nil
}

func WorkspaceSymbolsTestTypeToMatcher(typ WorkspaceSymbolsTestType) source.SymbolMatcher {
	switch typ {
	case WorkspaceSymbolsFuzzy:
		return source.SymbolFuzzy
	case WorkspaceSymbolsCaseSensitive:
		return source.SymbolCaseSensitive
	default:
		return source.SymbolCaseInsensitive
	}
}

func Diff(t *testing.T, want, got string) string {
	if want == got {
		return ""
	}
	// Add newlines to avoid newline messages in diff.
	want += "\n"
	got += "\n"
	d, err := myers.ComputeEdits("", want, got)
	if err != nil {
		t.Fatal(err)
	}
	return fmt.Sprintf("%q", diff.ToUnified("want", "got", want, d))
}

// StripSubscripts removes type parameter id subscripts.
//
// TODO(rfindley): remove this function once subscripts are removed from the
// type parameter type string.
func StripSubscripts(s string) string {
	var runes []rune
	for _, r := range s {
		// For debugging/uniqueness purposes, TypeString on a type parameter adds a
		// subscript corresponding to the type parameter's unique id. This is going
		// to be removed, but in the meantime we skip the subscript runes to get a
		// deterministic output.
		if '₀' <= r && r < '₀'+10 {
			continue // trim type parameter subscripts
		}
		runes = append(runes, r)
	}
	return string(runes)
}
