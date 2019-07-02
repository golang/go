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
		results, err := source.Diagnostics(context.Background(), r.view, f.(source.GoFile), nil)
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
		list, surrounding, err := source.Completion(ctx, r.view, f.(source.GoFile), pos, source.CompletionOptions{
			DeepComplete: strings.Contains(string(src.URI()), "deepcomplete"),
		})
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}
		var prefix string
		if surrounding != nil {
			prefix = strings.ToLower(surrounding.Prefix())
		}
		wantBuiltins := strings.Contains(string(src.URI()), "builtins")
		var got []source.CompletionItem
		for _, item := range list {
			if !wantBuiltins && isBuiltin(item) {
				continue
			}
			// We let the client do fuzzy matching, so we return all possible candidates.
			// To simplify testing, filter results with prefixes that don't match exactly.
			if !strings.HasPrefix(strings.ToLower(item.Label), prefix) {
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
			list, _, err := source.Completion(ctx, r.view, f.(source.GoFile), pos, source.CompletionOptions{
				DeepComplete: strings.Contains(string(src.URI()), "deepcomplete"),
			})
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
	sort.SliceStable(got, func(i, j int) bool {
		return got[i].Score > got[j].Score
	})

	// duplicate the lsp/completion logic to limit deep candidates to keep expected
	// list short
	var idx, seenDeepCompletions int
	for _, item := range got {
		if item.Depth > 0 {
			if seenDeepCompletions >= 3 {
				continue
			}
			seenDeepCompletions++
		}
		got[idx] = item
		idx++
	}
	got = got[:idx]

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
		filename := uri.Filename()
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
		data, _, err := f.Handle(ctx).Read(ctx)
		if err != nil {
			t.Error(err)
			continue
		}
		got := strings.Join(diff.ApplyEdits(diff.SplitLines(string(data)), ops), "")
		if gofmted != got {
			t.Errorf("format failed for %s, expected:\n%v\ngot:\n%v", filename, gofmted, got)
		}
	}
}

func (r *runner) Import(t *testing.T, data tests.Imports) {
	ctx := context.Background()
	for _, spn := range data {
		uri := spn.URI()
		filename := uri.Filename()
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
		edits, err := source.Imports(ctx, r.view, f.(source.GoFile), rng)
		if err != nil {
			if goimported != "" {
				t.Error(err)
			}
			continue
		}
		ops := source.EditsToDiff(edits)
		data, _, err := f.Handle(ctx).Read(ctx)
		if err != nil {
			t.Error(err)
			continue
		}
		got := strings.Join(diff.ApplyEdits(diff.SplitLines(string(data)), ops), "")
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
		hover, err := ident.Hover(ctx, false, source.SynopsisDocumentation)
		if err != nil {
			t.Fatalf("failed for %v: %v", d.Src, err)
		}
		rng := ident.DeclarationRange()
		if d.IsType {
			rng = ident.Type.Range
			hover = ""
		}
		if hover != "" {
			tag := fmt.Sprintf("%s-hover", d.Name)
			expectHover := string(r.data.Golden(tag, d.Src.URI().Filename(), func() ([]byte, error) {
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
		highlights, err := source.Highlight(ctx, f.(source.GoFile), pos)
		if err != nil {
			t.Errorf("highlight failed for %s: %v", src.URI(), err)
		}
		if len(highlights) != len(locations) {
			t.Errorf("got %d highlights for %s, expected %d", len(highlights), name, len(locations))
		}
		for i, h := range highlights {
			if h != locations[i] {
				t.Errorf("want %v, got %v\n", locations[i], h)
			}
		}
	}
}

func (r *runner) Reference(t *testing.T, data tests.References) {
	ctx := context.Background()
	for src, itemList := range data {
		f, err := r.view.GetFile(ctx, src.URI())
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}

		tok := f.GetToken(ctx)
		pos := tok.Pos(src.Start().Offset())
		ident, err := source.Identifier(ctx, r.view, f.(source.GoFile), pos)
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}

		want := make(map[span.Span]bool)
		for _, pos := range itemList {
			want[pos] = true
		}

		refs, err := ident.References(ctx)
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}

		got := make(map[span.Span]bool)
		for _, refInfo := range refs {
			refSpan, err := refInfo.Range.Span()
			if err != nil {
				t.Errorf("failed for %v item %v: %v", src, refInfo.Name, err)
			}
			got[refSpan] = true
		}

		if len(got) != len(want) {
			t.Errorf("references failed: different lengths got %v want %v", len(got), len(want))
		}

		for spn, _ := range got {
			if !want[spn] {
				t.Errorf("references failed: incorrect references got %v want locations %v", got, want)
			}
		}
	}
}

func (r *runner) Rename(t *testing.T, data tests.Renames) {
	ctx := context.Background()
	for spn, newText := range data {
		tag := fmt.Sprintf("%s-rename", newText)

		f, err := r.view.GetFile(ctx, spn.URI())
		if err != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}
		tok := f.GetToken(ctx)
		pos := tok.Pos(spn.Start().Offset())

		ident, err := source.Identifier(context.Background(), r.view, f.(source.GoFile), pos)
		if err != nil {
			t.Error(err)
		}
		changes, err := ident.Rename(context.Background(), newText)
		if err != nil {
			renamed := string(r.data.Golden(tag, spn.URI().Filename(), func() ([]byte, error) {
				return []byte(err.Error()), nil
			}))
			if err.Error() != renamed {
				t.Errorf("rename failed for %s, expected:\n%v\ngot:\n%v\n", newText, renamed, err)
			}
			continue
		}

		if len(changes) != 1 { // Renames must only affect a single file in these tests.
			t.Errorf("rename failed for %s, edited %d files, wanted 1 file", newText, len(changes))
			continue
		}

		edits := changes[spn.URI()]
		if edits == nil {
			t.Errorf("rename failed for %s, did not edit %s", newText, spn.URI())
			continue
		}
		data, _, err := f.Handle(ctx).Read(ctx)
		if err != nil {
			t.Error(err)
			continue
		}

		got := applyEdits(string(data), edits)
		gorenamed := string(r.data.Golden(tag, spn.URI().Filename(), func() ([]byte, error) {
			return []byte(got), nil
		}))

		if gorenamed != got {
			t.Errorf("rename failed for %s, expected:\n%v\ngot:\n%v", newText, gorenamed, got)
		}
	}
}

func applyEdits(contents string, edits []source.TextEdit) string {
	res := contents
	sortSourceTextEdits(edits)

	// Apply the edits from the end of the file forward
	// to preserve the offsets
	for i := len(edits) - 1; i >= 0; i-- {
		edit := edits[i]
		start := edit.Span.Start().Offset()
		end := edit.Span.End().Offset()
		tmp := res[0:start] + edit.NewText
		res = tmp + res[end:]
	}
	return res
}

func sortSourceTextEdits(d []source.TextEdit) {
	sort.Slice(d, func(i int, j int) bool {
		if r := span.Compare(d[i].Span, d[j].Span); r != 0 {
			return r < 0
		}
		return d[i].NewText < d[j].NewText
	})
}

func (r *runner) Symbol(t *testing.T, data tests.Symbols) {
	ctx := context.Background()
	for uri, expectedSymbols := range data {
		f, err := r.view.GetFile(ctx, uri)
		if err != nil {
			t.Fatalf("failed for %v: %v", uri, err)
		}
		symbols, err := source.DocumentSymbols(ctx, f.(source.GoFile))
		if err != nil {
			t.Errorf("symbols failed for %s: %v", uri, err)
		}
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
	for spn, expectedSignature := range data {
		f, err := r.view.GetFile(ctx, spn.URI())
		if err != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}
		tok := f.GetToken(ctx)
		pos := tok.Pos(spn.Start().Offset())
		gotSignature, err := source.SignatureHelp(ctx, f.(source.GoFile), pos)
		if err != nil {
			// Only fail if we got an error we did not expect.
			if expectedSignature != nil {
				t.Fatalf("failed for %v: %v", spn, err)
			}
		}
		if expectedSignature == nil {
			if gotSignature != nil {
				t.Errorf("expected no signature, got %v", gotSignature)
			}
			continue
		}
		if diff := diffSignatures(spn, expectedSignature, gotSignature); diff != "" {
			t.Error(diff)
		}
	}
}

func diffSignatures(spn span.Span, want *source.SignatureInformation, got *source.SignatureInformation) string {
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
	// This is a pure LSP feature, no source level functionality to be tested.
}
