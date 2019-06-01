// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source_test

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/lsp/xlog"
	"golang.org/x/tools/internal/span"
)

func TestSource(t *testing.T) {
	packagestest.TestAll(t, testSource)
}

type runner struct {
	view source.View
	data *tests.Data
}

func testSource(t *testing.T, exporter packagestest.Exporter) {
	data := tests.Load(t, exporter, "../testdata")
	defer data.Exported.Cleanup()

	log := xlog.New(xlog.StdSink{})
	cache := cache.New()
	session := cache.NewSession(log)
	r := &runner{
		view: session.NewView("source_test", span.FileURI(data.Config.Dir)),
		data: data,
	}
	r.view.SetEnv(data.Config.Env)
	for filename, content := range data.Config.Overlay {
		session.SetOverlay(span.FileURI(filename), content)
	}
	tests.Run(t, r, data)
}

func (r *runner) Diagnostics(t *testing.T, data tests.Diagnostics) {
	for uri, want := range data {
		f, err := r.view.GetFile(context.Background(), uri)
		if err != nil {
			t.Fatal(err)
		}
		results, err := source.Diagnostics(context.Background(), r.view, f.(source.GoFile))
		if err != nil {
			t.Fatal(err)
		}
		got := results[uri]
		if diff := diffDiagnostics(uri, want, got); diff != "" {
			t.Error(diff)
		}
	}
}

func sortDiagnostics(d []source.Diagnostic) {
	sort.Slice(d, func(i int, j int) bool {
		if r := span.Compare(d[i].Span, d[j].Span); r != 0 {
			return r < 0
		}
		return d[i].Message < d[j].Message
	})
}

// diffDiagnostics prints the diff between expected and actual diagnostics test
// results.
func diffDiagnostics(uri span.URI, want, got []source.Diagnostic) string {
	sortDiagnostics(want)
	sortDiagnostics(got)
	if len(got) != len(want) {
		return summarizeDiagnostics(-1, want, got, "different lengths got %v want %v", len(got), len(want))
	}
	for i, w := range want {
		g := got[i]
		if w.Message != g.Message {
			return summarizeDiagnostics(i, want, got, "incorrect Message got %v want %v", g.Message, w.Message)
		}
		if span.ComparePoint(w.Start(), g.Start()) != 0 {
			return summarizeDiagnostics(i, want, got, "incorrect Start got %v want %v", g.Start(), w.Start())
		}
		// Special case for diagnostics on parse errors.
		if strings.Contains(string(uri), "noparse") {
			if span.ComparePoint(g.Start(), g.End()) != 0 || span.ComparePoint(w.Start(), g.End()) != 0 {
				return summarizeDiagnostics(i, want, got, "incorrect End got %v want %v", g.End(), w.Start())
			}
		} else if !g.IsPoint() { // Accept any 'want' range if the diagnostic returns a zero-length range.
			if span.ComparePoint(w.End(), g.End()) != 0 {
				return summarizeDiagnostics(i, want, got, "incorrect End got %v want %v", g.End(), w.End())
			}
		}
		if w.Severity != g.Severity {
			return summarizeDiagnostics(i, want, got, "incorrect Severity got %v want %v", g.Severity, w.Severity)
		}
		if w.Source != g.Source {
			return summarizeDiagnostics(i, want, got, "incorrect Source got %v want %v", g.Source, w.Source)
		}
	}
	return ""
}

func summarizeDiagnostics(i int, want []source.Diagnostic, got []source.Diagnostic, reason string, args ...interface{}) string {
	msg := &bytes.Buffer{}
	fmt.Fprint(msg, "diagnostics failed")
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

func (r *runner) Completion(t *testing.T, data tests.Completions, snippets tests.CompletionSnippets, items tests.CompletionItems) {
	ctx := context.Background()
	for src, itemList := range data {
		var want []source.CompletionItem
		for _, pos := range itemList {
			want = append(want, *items[pos])
		}
		f, err := r.view.GetFile(ctx, src.URI())
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}
		tok := f.(source.GoFile).GetToken(ctx)
		if tok == nil {
			t.Fatalf("failed to get token for %v", src)
		}
		pos := tok.Pos(src.Start().Offset())
		list, surrounding, err := source.Completion(ctx, f.(source.GoFile), pos)
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}
		wantBuiltins := strings.Contains(string(src.URI()), "builtins")
		var got []source.CompletionItem
		for _, item := range list {
			if !wantBuiltins && isBuiltin(item) {
				continue
			}
			var prefix string
			if surrounding != nil {
				prefix = surrounding.Prefix()
			}
			// We let the client do fuzzy matching, so we return all possible candidates.
			// To simplify testing, filter results with prefixes that don't match exactly.
			if !strings.HasPrefix(item.Label, prefix) {
				continue
			}
			got = append(got, item)
		}
		if diff := diffCompletionItems(t, src, want, got); diff != "" {
			t.Errorf("%s: %s", src, diff)
		}
	}
	for _, usePlaceholders := range []bool{true, false} {
		for src, want := range snippets {
			f, err := r.view.GetFile(ctx, src.URI())
			if err != nil {
				t.Fatalf("failed for %v: %v", src, err)
			}
			tok := f.GetToken(ctx)
			pos := tok.Pos(src.Start().Offset())
			list, _, err := source.Completion(ctx, f.(source.GoFile), pos)
			if err != nil {
				t.Fatalf("failed for %v: %v", src, err)
			}
			wantItem := items[want.CompletionItem]
			var got *source.CompletionItem
			for _, item := range list {
				if item.Label == wantItem.Label {
					got = &item
					break
				}
			}
			if got == nil {
				t.Fatalf("%s: couldn't find completion matching %q", src.URI(), wantItem.Label)
			}
			expected := want.PlainSnippet
			if usePlaceholders {
				expected = want.PlaceholderSnippet
			}
			if actual := got.Snippet(usePlaceholders); expected != actual {
				t.Errorf("%s: expected placeholder snippet %q, got %q", src, expected, actual)
			}
		}
	}
}

func isBuiltin(item source.CompletionItem) bool {
	// If a type has no detail, it is a builtin type.
	if item.Detail == "" && item.Kind == source.TypeCompletionItem {
		return true
	}
	// Remaining builtin constants, variables, interfaces, and functions.
	trimmed := item.Label
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

// diffCompletionItems prints the diff between expected and actual completion
// test results.
func diffCompletionItems(t *testing.T, spn span.Span, want []source.CompletionItem, got []source.CompletionItem) string {
	if len(got) != len(want) {
		return summarizeCompletionItems(-1, want, got, "different lengths got %v want %v", len(got), len(want))
	}
	sort.SliceStable(got, func(i, j int) bool {
		return got[i].Score > got[j].Score
	})
	for i, w := range want {
		g := got[i]
		if w.Label != g.Label {
			return summarizeCompletionItems(i, want, got, "incorrect Label got %v want %v", g.Label, w.Label)
		}
		if w.Detail != g.Detail {
			return summarizeCompletionItems(i, want, got, "incorrect Detail got %v want %v", g.Detail, w.Detail)
		}
		if w.Kind != g.Kind {
			return summarizeCompletionItems(i, want, got, "incorrect Kind got %v want %v", g.Kind, w.Kind)
		}
	}
	return ""
}

func summarizeCompletionItems(i int, want []source.CompletionItem, got []source.CompletionItem, reason string, args ...interface{}) string {
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

func (r *runner) Format(t *testing.T, data tests.Formats) {
	ctx := context.Background()
	for _, spn := range data {
		uri := spn.URI()
		filename, err := uri.Filename()
		if err != nil {
			t.Fatal(err)
		}
		gofmted := string(r.data.Golden("gofmt", filename, func() ([]byte, error) {
			cmd := exec.Command("gofmt", filename)
			out, _ := cmd.Output() // ignore error, sometimes we have intentionally ungofmt-able files
			return out, nil
		}))
		f, err := r.view.GetFile(ctx, uri)
		if err != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}
		rng, err := spn.Range(span.NewTokenConverter(f.FileSet(), f.GetToken(ctx)))
		if err != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}
		edits, err := source.Format(ctx, f.(source.GoFile), rng)
		if err != nil {
			if gofmted != "" {
				t.Error(err)
			}
			continue
		}
		ops := source.EditsToDiff(edits)
		fc := f.Content(ctx)
		if fc.Error != nil {
			t.Error(err)
			continue
		}
		got := strings.Join(diff.ApplyEdits(diff.SplitLines(string(fc.Data)), ops), "")
		if gofmted != got {
			t.Errorf("format failed for %s, expected:\n%v\ngot:\n%v", filename, gofmted, got)
		}
	}
}

func (r *runner) Import(t *testing.T, data tests.Imports) {
	ctx := context.Background()
	for _, spn := range data {
		uri := spn.URI()
		filename, err := uri.Filename()
		if err != nil {
			t.Fatal(err)
		}
		goimported := string(r.data.Golden("goimports", filename, func() ([]byte, error) {
			cmd := exec.Command("goimports", filename)
			out, _ := cmd.Output() // ignore error, sometimes we have intentionally ungofmt-able files
			return out, nil
		}))
		f, err := r.view.GetFile(ctx, uri)
		if err != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}
		rng, err := spn.Range(span.NewTokenConverter(f.FileSet(), f.GetToken(ctx)))
		if err != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}
		edits, err := source.Imports(ctx, f.(source.GoFile), rng)
		if err != nil {
			if goimported != "" {
				t.Error(err)
			}
			continue
		}
		ops := source.EditsToDiff(edits)
		fc := f.Content(ctx)
		if fc.Error != nil {
			t.Error(err)
			continue
		}
		got := strings.Join(diff.ApplyEdits(diff.SplitLines(string(fc.Data)), ops), "")
		if goimported != got {
			t.Errorf("import failed for %s, expected:\n%v\ngot:\n%v", filename, goimported, got)
		}
	}
}

func (r *runner) Definition(t *testing.T, data tests.Definitions) {
	ctx := context.Background()
	for _, d := range data {
		f, err := r.view.GetFile(ctx, d.Src.URI())
		if err != nil {
			t.Fatalf("failed for %v: %v", d.Src, err)
		}
		tok := f.GetToken(ctx)
		pos := tok.Pos(d.Src.Start().Offset())
		ident, err := source.Identifier(ctx, r.view, f.(source.GoFile), pos)
		if err != nil {
			t.Fatalf("failed for %v: %v", d.Src, err)
		}
		hover, err := ident.Hover(ctx, nil, false, true)
		if err != nil {
			t.Fatalf("failed for %v: %v", d.Src, err)
		}
		rng := ident.Declaration.Range
		if d.IsType {
			rng = ident.Type.Range
			hover = ""
		}
		if hover != "" {
			tag := fmt.Sprintf("%s-hover", d.Name)
			filename, err := d.Src.URI().Filename()
			if err != nil {
				t.Fatalf("failed for %v: %v", d.Def, err)
			}
			expectHover := string(r.data.Golden(tag, filename, func() ([]byte, error) {
				return []byte(hover), nil
			}))
			if hover != expectHover {
				t.Errorf("for %v got %q want %q", d.Src, hover, expectHover)
			}
		} else if !d.OnlyHover {
			if def, err := rng.Span(); err != nil {
				t.Fatalf("failed for %v: %v", rng, err)
			} else if def != d.Def {
				t.Errorf("for %v got %v want %v", d.Src, def, d.Def)
			}
		} else {
			t.Errorf("no tests ran for %s", d.Src.URI())
		}
	}
}

func (r *runner) Highlight(t *testing.T, data tests.Highlights) {
	ctx := context.Background()
	for name, locations := range data {
		src := locations[0]
		f, err := r.view.GetFile(ctx, src.URI())
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}
		tok := f.GetToken(ctx)
		pos := tok.Pos(src.Start().Offset())
		highlights := source.Highlight(ctx, f.(source.GoFile), pos)
		if len(highlights) != len(locations) {
			t.Fatalf("got %d highlights for %s, expected %d", len(highlights), name, len(locations))
		}
		for i, h := range highlights {
			if h != locations[i] {
				t.Errorf("want %v, got %v\n", locations[i], h)
			}
		}
	}
}

func (r *runner) Symbol(t *testing.T, data tests.Symbols) {
	ctx := context.Background()
	for uri, expectedSymbols := range data {
		f, err := r.view.GetFile(ctx, uri)
		if err != nil {
			t.Fatalf("failed for %v: %v", uri, err)
		}
		symbols := source.DocumentSymbols(ctx, f.(source.GoFile))

		if len(symbols) != len(expectedSymbols) {
			t.Errorf("want %d top-level symbols in %v, got %d", len(expectedSymbols), uri, len(symbols))
			continue
		}
		if diff := r.diffSymbols(uri, expectedSymbols, symbols); diff != "" {
			t.Error(diff)
		}
	}
}

func (r *runner) diffSymbols(uri span.URI, want []source.Symbol, got []source.Symbol) string {
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
		if w.SelectionSpan != g.SelectionSpan {
			return summarizeSymbols(i, want, got, "incorrect span got %v want %v", g.SelectionSpan, w.SelectionSpan)
		}
		if msg := r.diffSymbols(uri, w.Children, g.Children); msg != "" {
			return fmt.Sprintf("children of %s: %s", w.Name, msg)
		}
	}
	return ""
}

func summarizeSymbols(i int, want []source.Symbol, got []source.Symbol, reason string, args ...interface{}) string {
	msg := &bytes.Buffer{}
	fmt.Fprint(msg, "document symbols failed")
	if i >= 0 {
		fmt.Fprintf(msg, " at %d", i)
	}
	fmt.Fprint(msg, " because of ")
	fmt.Fprintf(msg, reason, args...)
	fmt.Fprint(msg, ":\nexpected:\n")
	for _, s := range want {
		fmt.Fprintf(msg, "  %v %v %v\n", s.Name, s.Kind, s.SelectionSpan)
	}
	fmt.Fprintf(msg, "got:\n")
	for _, s := range got {
		fmt.Fprintf(msg, "  %v %v %v\n", s.Name, s.Kind, s.SelectionSpan)
	}
	return msg.String()
}

func (r *runner) SignatureHelp(t *testing.T, data tests.Signatures) {
	ctx := context.Background()
	for spn, expectedSignatures := range data {
		f, err := r.view.GetFile(ctx, spn.URI())
		if err != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}
		tok := f.GetToken(ctx)
		pos := tok.Pos(spn.Start().Offset())
		gotSignature, err := source.SignatureHelp(ctx, f.(source.GoFile), pos)
		if err != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}
		if diff := diffSignatures(spn, expectedSignatures, *gotSignature); diff != "" {
			t.Error(diff)
		}
	}
}

func diffSignatures(spn span.Span, want source.SignatureInformation, got source.SignatureInformation) string {
	decorate := func(f string, args ...interface{}) string {
		return fmt.Sprintf("Invalid signature at %s: %s", spn, fmt.Sprintf(f, args...))
	}

	if want.ActiveParameter != got.ActiveParameter {
		return decorate("wanted active parameter of %d, got %f", want.ActiveParameter, got.ActiveParameter)
	}

	if want.Label != got.Label {
		return decorate("wanted label %q, got %q", want.Label, got.Label)
	}

	var paramParts []string
	for _, p := range got.Parameters {
		paramParts = append(paramParts, p.Label)
	}
	paramsStr := strings.Join(paramParts, ", ")
	if !strings.Contains(got.Label, paramsStr) {
		return decorate("expected signature %q to contain params %q", got.Label, paramsStr)
	}

	return ""
}

func (r *runner) Link(t *testing.T, data tests.Links) {
	//This is a pure LSP feature, no source level functionality to be tested
}
