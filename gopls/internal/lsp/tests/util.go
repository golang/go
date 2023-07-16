// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tests

import (
	"bytes"
	"fmt"
	"go/token"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/source/completion"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
	"golang.org/x/tools/gopls/internal/span"
)

var builtins = map[string]bool{
	"append":  true,
	"cap":     true,
	"close":   true,
	"complex": true,
	"copy":    true,
	"delete":  true,
	"error":   true,
	"false":   true,
	"imag":    true,
	"iota":    true,
	"len":     true,
	"make":    true,
	"new":     true,
	"nil":     true,
	"panic":   true,
	"print":   true,
	"println": true,
	"real":    true,
	"recover": true,
	"true":    true,
}

// DiffLinks takes the links we got and checks if they are located within the source or a Note.
// If the link is within a Note, the link is removed.
// Returns an diff comment if there are differences and empty string if no diffs.
func DiffLinks(mapper *protocol.Mapper, wantLinks []Link, gotLinks []protocol.DocumentLink) string {
	var notePositions []token.Position
	links := make(map[span.Span]string, len(wantLinks))
	for _, link := range wantLinks {
		links[link.Src] = link.Target
		notePositions = append(notePositions, link.NotePosition)
	}

	var msg strings.Builder
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
			if target != *link.Target {
				fmt.Fprintf(&msg, "%s: want link with target %q, got %q\n", spn, target, *link.Target)
			}
		} else {
			fmt.Fprintf(&msg, "%s: got unexpected link with target %q\n", spn, *link.Target)
		}
	}
	for spn, target := range links {
		fmt.Fprintf(&msg, "%s: expected link with target %q is missing\n", spn, target)
	}
	return msg.String()
}

// CompareDiagnostics reports testing errors to t when the diagnostic set got
// does not match want.
func CompareDiagnostics(t *testing.T, uri span.URI, want, got []*source.Diagnostic) {
	t.Helper()
	fileName := path.Base(string(uri))

	// Build a helper function to match an actual diagnostic to an overlapping
	// expected diagnostic (if any).
	unmatched := make([]*source.Diagnostic, len(want))
	copy(unmatched, want)
	source.SortDiagnostics(unmatched)
	match := func(g *source.Diagnostic) *source.Diagnostic {
		// Find the last expected diagnostic d for which start(d) < end(g), and
		// check to see if it overlaps.
		i := sort.Search(len(unmatched), func(i int) bool {
			d := unmatched[i]
			// See rangeOverlaps: if a range is a single point, we consider End to be
			// included in the range...
			if g.Range.Start == g.Range.End {
				return protocol.ComparePosition(d.Range.Start, g.Range.End) > 0
			}
			// ...otherwise the end position of a range is not included.
			return protocol.ComparePosition(d.Range.Start, g.Range.End) >= 0
		})
		if i == 0 {
			return nil
		}
		w := unmatched[i-1]
		if rangeOverlaps(w.Range, g.Range) {
			unmatched = append(unmatched[:i-1], unmatched[i:]...)
			return w
		}
		return nil
	}

	for _, g := range got {
		w := match(g)
		if w == nil {
			t.Errorf("%s:%s: unexpected diagnostic %q", fileName, g.Range, g.Message)
			continue
		}
		if match, err := regexp.MatchString(w.Message, g.Message); err != nil {
			t.Errorf("%s:%s: invalid regular expression %q: %v", fileName, w.Range.Start, w.Message, err)
		} else if !match {
			t.Errorf("%s:%s: got Message %q, want match for pattern %q", fileName, g.Range.Start, g.Message, w.Message)
		}
		if w.Severity != g.Severity {
			t.Errorf("%s:%s: got Severity %v, want %v", fileName, g.Range.Start, g.Severity, w.Severity)
		}
		if w.Source != g.Source {
			t.Errorf("%s:%s: got Source %v, want %v", fileName, g.Range.Start, g.Source, w.Source)
		}
	}

	for _, w := range unmatched {
		t.Errorf("%s:%s: unmatched diagnostic pattern %q", fileName, w.Range, w.Message)
	}
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
	if diff := compare.Text(NormalizeAny(w.Label), NormalizeAny(g.Label)); diff != "" {
		return decorate("mismatched labels:\n%s", diff)
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
	return builtins[trimmed]
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
//
// The diff will be formatted using '-' and '+' for want and got, respectively.
func DiffCompletionItems(want, got []protocol.CompletionItem) string {
	// Many fields are not set in the "want" slice.
	irrelevantFields := []string{
		"AdditionalTextEdits",
		"Documentation",
		"TextEdit",
		"SortText",
		"Preselect",
		"FilterText",
		"InsertText",
		"InsertTextFormat",
	}
	ignore := cmpopts.IgnoreFields(protocol.CompletionItem{}, irrelevantFields...)
	normalizeAny := cmpopts.AcyclicTransformer("NormalizeAny", func(item protocol.CompletionItem) protocol.CompletionItem {
		item.Detail = NormalizeAny(item.Detail)
		return item
	})
	return cmp.Diff(want, got, ignore, normalizeAny)
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

func EnableAllAnalyzers(opts *source.Options) {
	if opts.Analyses == nil {
		opts.Analyses = make(map[string]bool)
	}
	for _, a := range opts.DefaultAnalyzers {
		if !a.IsEnabled(opts) {
			opts.Analyses[a.Analyzer.Name] = true
		}
	}
	for _, a := range opts.TypeErrorAnalyzers {
		if !a.IsEnabled(opts) {
			opts.Analyses[a.Analyzer.Name] = true
		}
	}
	for _, a := range opts.ConvenienceAnalyzers {
		if !a.IsEnabled(opts) {
			opts.Analyses[a.Analyzer.Name] = true
		}
	}
	for _, a := range opts.StaticcheckAnalyzers {
		if !a.IsEnabled(opts) {
			opts.Analyses[a.Analyzer.Name] = true
		}
	}
}

func EnableAllInlayHints(opts *source.Options) {
	if opts.Hints == nil {
		opts.Hints = make(map[string]bool)
	}
	for name := range source.AllInlayHints {
		opts.Hints[name] = true
	}
}
