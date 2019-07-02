// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"fmt"
	"go/token"
	"os/exec"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/lsp/xlog"
	"golang.org/x/tools/internal/span"
)

func TestLSP(t *testing.T) {
	packagestest.TestAll(t, testLSP)
}

type runner struct {
	server *Server
	data   *tests.Data
}

const viewName = "lsp_test"

func testLSP(t *testing.T, exporter packagestest.Exporter) {
	data := tests.Load(t, exporter, "testdata")
	defer data.Exported.Cleanup()

	log := xlog.New(xlog.StdSink{})
	cache := cache.New()
	session := cache.NewSession(log)
	view := session.NewView(viewName, span.FileURI(data.Config.Dir))
	view.SetEnv(data.Config.Env)
	for filename, content := range data.Config.Overlay {
		session.SetOverlay(span.FileURI(filename), content)
	}
	r := &runner{
		server: &Server{
			session:     session,
			undelivered: make(map[span.URI][]source.Diagnostic),
			supportedCodeActions: map[protocol.CodeActionKind]bool{
				protocol.SourceOrganizeImports: true,
				protocol.QuickFix:              true,
			},
			hoverKind: source.SynopsisDocumentation,
		},
		data: data,
	}
	tests.Run(t, r, data)
}

// TODO: Actually test the LSP diagnostics function in this test.
func (r *runner) Diagnostics(t *testing.T, data tests.Diagnostics) {
	v := r.server.session.View(viewName)
	for uri, want := range data {
		f, err := v.GetFile(context.Background(), uri)
		if err != nil {
			t.Fatalf("no file for %s: %v", f, err)
		}
		gof, ok := f.(source.GoFile)
		if !ok {
			t.Fatalf("%s is not a Go file: %v", uri, err)
		}
		results, err := source.Diagnostics(context.Background(), v, gof, nil)
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
		fmt.Fprintf(msg, "  %v: %s\n", d.Span, d.Message)
	}
	fmt.Fprintf(msg, "got:\n")
	for _, d := range got {
		fmt.Fprintf(msg, "  %v: %s\n", d.Span, d.Message)
	}
	return msg.String()
}

func (r *runner) Completion(t *testing.T, data tests.Completions, snippets tests.CompletionSnippets, items tests.CompletionItems) {
	defer func() { r.server.useDeepCompletions = false }()

	for src, itemList := range data {
		var want []source.CompletionItem
		for _, pos := range itemList {
			want = append(want, *items[pos])
		}

		r.server.useDeepCompletions = strings.Contains(string(src.URI()), "deepcomplete")

		list := r.runCompletion(t, src)

		wantBuiltins := strings.Contains(string(src.URI()), "builtins")
		var got []protocol.CompletionItem
		for _, item := range list.Items {
			if !wantBuiltins && isBuiltin(item) {
				continue
			}
			got = append(got, item)
		}
		if diff := diffCompletionItems(t, src, want, got); diff != "" {
			t.Errorf("%s: %s", src, diff)
		}
	}

	origPlaceHolders := r.server.usePlaceholders
	origTextFormat := r.server.insertTextFormat
	defer func() {
		r.server.usePlaceholders = origPlaceHolders
		r.server.insertTextFormat = origTextFormat
	}()

	r.server.insertTextFormat = protocol.SnippetTextFormat
	for _, usePlaceholders := range []bool{true, false} {
		r.server.usePlaceholders = usePlaceholders

		for src, want := range snippets {
			r.server.useDeepCompletions = strings.Contains(string(src.URI()), "deepcomplete")

			list := r.runCompletion(t, src)

			wantItem := items[want.CompletionItem]
			var got *protocol.CompletionItem
			for _, item := range list.Items {
				if item.Label == wantItem.Label {
					got = &item
					break
				}
			}
			if got == nil {
				t.Fatalf("%s: couldn't find completion matching %q", src.URI(), wantItem.Label)
			}
			var expected string
			if usePlaceholders {
				expected = want.PlaceholderSnippet
			} else {
				expected = want.PlainSnippet
			}
			if expected != got.TextEdit.NewText {
				t.Errorf("%s: expected snippet %q, got %q", src, expected, got.TextEdit.NewText)
			}
		}
	}
}

func (r *runner) runCompletion(t *testing.T, src span.Span) *protocol.CompletionList {
	t.Helper()
	list, err := r.server.Completion(context.Background(), &protocol.CompletionParams{
		TextDocumentPositionParams: protocol.TextDocumentPositionParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: protocol.NewURI(src.URI()),
			},
			Position: protocol.Position{
				Line:      float64(src.Start().Line() - 1),
				Character: float64(src.Start().Column() - 1),
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return list
}

func isBuiltin(item protocol.CompletionItem) bool {
	// If a type has no detail, it is a builtin type.
	if item.Detail == "" && item.Kind == protocol.TypeParameterCompletion {
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
func diffCompletionItems(t *testing.T, spn span.Span, want []source.CompletionItem, got []protocol.CompletionItem) string {
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
		if wkind := toProtocolCompletionItemKind(w.Kind); wkind != g.Kind {
			return summarizeCompletionItems(i, want, got, "incorrect Kind got %v want %v", g.Kind, wkind)
		}
	}
	return ""
}

func summarizeCompletionItems(i int, want []source.CompletionItem, got []protocol.CompletionItem, reason string, args ...interface{}) string {
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

		edits, err := r.server.Formatting(context.Background(), &protocol.DocumentFormattingParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: protocol.NewURI(uri),
			},
		})
		if err != nil {
			if gofmted != "" {
				t.Error(err)
			}
			continue
		}
		_, m, err := getSourceFile(ctx, r.server.session.ViewOf(uri), uri)
		if err != nil {
			t.Error(err)
		}
		sedits, err := FromProtocolEdits(m, edits)
		if err != nil {
			t.Error(err)
		}
		ops := source.EditsToDiff(sedits)
		got := strings.Join(diff.ApplyEdits(diff.SplitLines(string(m.Content)), ops), "")
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

		actions, err := r.server.CodeAction(context.Background(), &protocol.CodeActionParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: protocol.NewURI(uri),
			},
		})
		if err != nil {
			if goimported != "" {
				t.Error(err)
			}
			continue
		}
		_, m, err := getSourceFile(ctx, r.server.session.ViewOf(uri), uri)
		if err != nil {
			t.Error(err)
		}
		var edits []protocol.TextEdit
		for _, a := range actions {
			if a.Title == "Organize Imports" {
				edits = (*a.Edit.Changes)[string(uri)]
			}
		}
		sedits, err := FromProtocolEdits(m, edits)
		if err != nil {
			t.Error(err)
		}
		ops := source.EditsToDiff(sedits)
		got := strings.Join(diff.ApplyEdits(diff.SplitLines(string(m.Content)), ops), "")
		if goimported != got {
			t.Errorf("import failed for %s, expected:\n%v\ngot:\n%v", filename, goimported, got)
		}
	}
}

func (r *runner) Definition(t *testing.T, data tests.Definitions) {
	for _, d := range data {
		sm, err := r.mapper(d.Src.URI())
		if err != nil {
			t.Fatal(err)
		}
		loc, err := sm.Location(d.Src)
		if err != nil {
			t.Fatalf("failed for %v: %v", d.Src, err)
		}
		params := &protocol.TextDocumentPositionParams{
			TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
			Position:     loc.Range.Start,
		}
		var locs []protocol.Location
		var hover *protocol.Hover
		if d.IsType {
			locs, err = r.server.TypeDefinition(context.Background(), params)
		} else {
			locs, err = r.server.Definition(context.Background(), params)
			if err != nil {
				t.Fatalf("failed for %v: %v", d.Src, err)
			}
			hover, err = r.server.Hover(context.Background(), params)
		}
		if err != nil {
			t.Fatalf("failed for %v: %v", d.Src, err)
		}
		if len(locs) != 1 {
			t.Errorf("got %d locations for definition, expected 1", len(locs))
		}
		if hover != nil {
			tag := fmt.Sprintf("%s-hover", d.Name)
			expectHover := string(r.data.Golden(tag, d.Src.URI().Filename(), func() ([]byte, error) {
				return []byte(hover.Contents.Value), nil
			}))
			if hover.Contents.Value != expectHover {
				t.Errorf("for %v got %q want %q", d.Src, hover.Contents.Value, expectHover)
			}
		} else if !d.OnlyHover {
			locURI := span.NewURI(locs[0].URI)
			lm, err := r.mapper(locURI)
			if err != nil {
				t.Fatal(err)
			}
			if def, err := lm.Span(locs[0]); err != nil {
				t.Fatalf("failed for %v: %v", locs[0], err)
			} else if def != d.Def {
				t.Errorf("for %v got %v want %v", d.Src, def, d.Def)
			}
		} else {
			t.Errorf("no tests ran for %s", d.Src.URI())
		}
	}
}

func (r *runner) Highlight(t *testing.T, data tests.Highlights) {
	for name, locations := range data {
		m, err := r.mapper(locations[0].URI())
		if err != nil {
			t.Fatal(err)
		}
		loc, err := m.Location(locations[0])
		if err != nil {
			t.Fatalf("failed for %v: %v", locations[0], err)
		}
		params := &protocol.TextDocumentPositionParams{
			TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
			Position:     loc.Range.Start,
		}
		highlights, err := r.server.DocumentHighlight(context.Background(), params)
		if err != nil {
			t.Fatal(err)
		}
		if len(highlights) != len(locations) {
			t.Fatalf("got %d highlights for %s, expected %d", len(highlights), name, len(locations))
		}
		for i := range highlights {
			if h, err := m.RangeSpan(highlights[i].Range); err != nil {
				t.Fatalf("failed for %v: %v", highlights[i], err)
			} else if h != locations[i] {
				t.Errorf("want %v, got %v\n", locations[i], h)
			}
		}
	}
}

func (r *runner) Reference(t *testing.T, data tests.References) {
	for src, itemList := range data {
		sm, err := r.mapper(src.URI())
		if err != nil {
			t.Fatal(err)
		}
		loc, err := sm.Location(src)
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}

		want := make(map[protocol.Location]bool)
		for _, pos := range itemList {
			m, err := r.mapper(pos.URI())
			if err != nil {
				t.Fatal(err)
			}
			loc, err := m.Location(pos)
			if err != nil {
				t.Fatalf("failed for %v: %v", src, err)
			}
			want[loc] = true
		}
		params := &protocol.ReferenceParams{
			TextDocumentPositionParams: protocol.TextDocumentPositionParams{
				TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
				Position:     loc.Range.Start,
			},
		}
		got, err := r.server.References(context.Background(), params)
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}

		if len(got) != len(want) {
			t.Errorf("references failed: different lengths got %v want %v", len(got), len(want))
		}
		for _, loc := range got {
			if !want[loc] {
				t.Errorf("references failed: incorrect references got %v want %v", loc, want)
			}
		}
	}
}

func (r *runner) Rename(t *testing.T, data tests.Renames) {
	ctx := context.Background()
	for spn, newText := range data {
		tag := fmt.Sprintf("%s-rename", newText)

		uri := spn.URI()
		filename := uri.Filename()
		sm, err := r.mapper(uri)
		if err != nil {
			t.Fatal(err)
		}
		loc, err := sm.Location(spn)
		if err != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}

		workspaceEdits, err := r.server.Rename(ctx, &protocol.RenameParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: protocol.NewURI(uri),
			},
			Position: loc.Range.Start,
			NewName:  newText,
		})
		if err != nil {
			renamed := string(r.data.Golden(tag, filename, func() ([]byte, error) {
				return []byte(err.Error()), nil
			}))
			if err.Error() != renamed {
				t.Errorf("rename failed for %s, expected:\n%v\ngot:\n%v\n", newText, renamed, err)
			}
			continue
		}

		_, m, err := getSourceFile(ctx, r.server.session.ViewOf(uri), uri)
		if err != nil {
			t.Error(err)
		}

		changes := *workspaceEdits.Changes
		if len(changes) != 1 { // Renames must only affect a single file in these tests.
			t.Errorf("rename failed for %s, edited %d files, wanted 1 file", newText, len(*workspaceEdits.Changes))
			continue
		}

		edits := changes[string(uri)]
		if edits == nil {
			t.Errorf("rename failed for %s, did not edit %s", newText, filename)
			continue
		}
		sedits, err := FromProtocolEdits(m, edits)
		if err != nil {
			t.Error(err)
		}

		got := applyEdits(string(m.Content), sedits)

		gorenamed := string(r.data.Golden(tag, filename, func() ([]byte, error) {
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
	for uri, expectedSymbols := range data {
		params := &protocol.DocumentSymbolParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: string(uri),
			},
		}
		symbols, err := r.server.DocumentSymbol(context.Background(), params)
		if err != nil {
			t.Fatal(err)
		}

		if len(symbols) != len(expectedSymbols) {
			t.Errorf("want %d top-level symbols in %v, got %d", len(expectedSymbols), uri, len(symbols))
			continue
		}
		if diff := r.diffSymbols(t, uri, expectedSymbols, symbols); diff != "" {
			t.Error(diff)
		}
	}
}

func (r *runner) diffSymbols(t *testing.T, uri span.URI, want []source.Symbol, got []protocol.DocumentSymbol) string {
	sort.Slice(want, func(i, j int) bool { return want[i].Name < want[j].Name })
	sort.Slice(got, func(i, j int) bool { return got[i].Name < got[j].Name })
	m, err := r.mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		return summarizeSymbols(-1, want, got, "different lengths got %v want %v", len(got), len(want))
	}
	for i, w := range want {
		g := got[i]
		if w.Name != g.Name {
			return summarizeSymbols(i, want, got, "incorrect name got %v want %v", g.Name, w.Name)
		}
		if wkind := toProtocolSymbolKind(w.Kind); wkind != g.Kind {
			return summarizeSymbols(i, want, got, "incorrect kind got %v want %v", g.Kind, wkind)
		}
		spn, err := m.RangeSpan(g.SelectionRange)
		if err != nil {
			return summarizeSymbols(i, want, got, "%v", err)
		}
		if w.SelectionSpan != spn {
			return summarizeSymbols(i, want, got, "incorrect span got %v want %v", spn, w.SelectionSpan)
		}
		if msg := r.diffSymbols(t, uri, w.Children, g.Children); msg != "" {
			return fmt.Sprintf("children of %s: %s", w.Name, msg)
		}
	}
	return ""
}

func summarizeSymbols(i int, want []source.Symbol, got []protocol.DocumentSymbol, reason string, args ...interface{}) string {
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
		fmt.Fprintf(msg, "  %v %v %v\n", s.Name, s.Kind, s.SelectionRange)
	}
	return msg.String()
}

func (r *runner) SignatureHelp(t *testing.T, data tests.Signatures) {
	for spn, expectedSignatures := range data {
		m, err := r.mapper(spn.URI())
		if err != nil {
			t.Fatal(err)
		}
		loc, err := m.Location(spn)
		if err != nil {
			t.Fatalf("failed for %v: %v", loc, err)
		}
		gotSignatures, err := r.server.SignatureHelp(context.Background(), &protocol.TextDocumentPositionParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: protocol.NewURI(spn.URI()),
			},
			Position: loc.Range.Start,
		})
		if err != nil {
			// Only fail if we got an error we did not expect.
			if expectedSignatures != nil {
				t.Fatal(err)
			}
			continue
		}
		if expectedSignatures == nil {
			if gotSignatures != nil {
				t.Errorf("expected no signature, got %v", gotSignatures)
			}
			continue
		}
		if diff := diffSignatures(spn, expectedSignatures, gotSignatures); diff != "" {
			t.Error(diff)
		}
	}
}

func diffSignatures(spn span.Span, want *source.SignatureInformation, got *protocol.SignatureHelp) string {
	decorate := func(f string, args ...interface{}) string {
		return fmt.Sprintf("Invalid signature at %s: %s", spn, fmt.Sprintf(f, args...))
	}

	if len(got.Signatures) != 1 {
		return decorate("wanted 1 signature, got %d", len(got.Signatures))
	}

	if got.ActiveSignature != 0 {
		return decorate("wanted active signature of 0, got %f", got.ActiveSignature)
	}

	if want.ActiveParameter != int(got.ActiveParameter) {
		return decorate("wanted active parameter of %d, got %f", want.ActiveParameter, got.ActiveParameter)
	}

	gotSig := got.Signatures[int(got.ActiveSignature)]

	if want.Label != gotSig.Label {
		return decorate("wanted label %q, got %q", want.Label, gotSig.Label)
	}

	var paramParts []string
	for _, p := range gotSig.Parameters {
		paramParts = append(paramParts, p.Label)
	}
	paramsStr := strings.Join(paramParts, ", ")
	if !strings.Contains(gotSig.Label, paramsStr) {
		return decorate("expected signature %q to contain params %q", gotSig.Label, paramsStr)
	}

	return ""
}

func (r *runner) Link(t *testing.T, data tests.Links) {
	for uri, wantLinks := range data {
		m, err := r.mapper(uri)
		if err != nil {
			t.Fatal(err)
		}
		gotLinks, err := r.server.DocumentLink(context.Background(), &protocol.DocumentLinkParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: protocol.NewURI(uri),
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		links := make(map[span.Span]string, len(wantLinks))
		for _, link := range wantLinks {
			links[link.Src] = link.Target
		}
		for _, link := range gotLinks {
			spn, err := m.RangeSpan(link.Range)
			if err != nil {
				t.Fatal(err)
			}
			if target, ok := links[spn]; ok {
				delete(links, spn)
				if target != link.Target {
					t.Errorf("for %v want %v, got %v\n", spn, link.Target, target)
				}
			} else {
				t.Errorf("unexpected link %v:%v\n", spn, link.Target)
			}
		}
		for spn, target := range links {
			t.Errorf("missing link %v:%v\n", spn, target)
		}
	}
}

func (r *runner) mapper(uri span.URI) (*protocol.ColumnMapper, error) {
	filename := uri.Filename()
	fset := r.data.Exported.ExpectFileSet
	var f *token.File
	fset.Iterate(func(check *token.File) bool {
		if check.Name() == filename {
			f = check
			return false
		}
		return true
	})
	if f == nil {
		return nil, fmt.Errorf("no token.File for %s", uri)
	}
	content, err := r.data.Exported.FileContents(f.Name())
	if err != nil {
		return nil, err
	}
	return protocol.NewColumnMapper(uri, filename, fset, f, content), nil
}

func TestBytesOffset(t *testing.T) {
	tests := []struct {
		text string
		pos  protocol.Position
		want int
	}{
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 0}, want: 0},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 1}, want: 1},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 2}, want: 1},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 3}, want: 5},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 4}, want: 6},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 5}, want: -1},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 0, Character: 3}, want: 3},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 0, Character: 4}, want: 3},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 1, Character: 0}, want: 4},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 1, Character: 3}, want: 7},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 1, Character: 4}, want: 7},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 2, Character: 0}, want: 8},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 2, Character: 1}, want: -1},
		{text: "aaa\nbbb\n\n", pos: protocol.Position{Line: 2, Character: 0}, want: 8},
	}

	for i, test := range tests {
		fname := fmt.Sprintf("test %d", i)
		fset := token.NewFileSet()
		f := fset.AddFile(fname, -1, len(test.text))
		f.SetLinesForContent([]byte(test.text))
		mapper := protocol.NewColumnMapper(span.FileURI(fname), fname, fset, f, []byte(test.text))
		got, err := mapper.Point(test.pos)
		if err != nil && test.want != -1 {
			t.Errorf("unexpected error: %v", err)
		}
		if err == nil && got.Offset() != test.want {
			t.Errorf("want %d for %q(Line:%d,Character:%d), but got %d", test.want, test.text, int(test.pos.Line), int(test.pos.Character), got.Offset())
		}
	}
}
