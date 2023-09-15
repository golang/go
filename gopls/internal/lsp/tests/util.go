// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tests

import (
	"bytes"
	"fmt"
	"go/token"
	"sort"
	"strconv"
	"strings"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
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
