// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/xlog"
	"golang.org/x/tools/internal/span"
)

// TODO(rstambler): Remove this once Go 1.12 is released as we end support for
// versions of Go <= 1.10.
var goVersion111 = true

func TestLSP(t *testing.T) {
	packagestest.TestAll(t, testLSP)
}

func testLSP(t *testing.T, exporter packagestest.Exporter) {
	ctx := context.Background()
	const dir = "testdata"

	// We hardcode the expected number of test cases to ensure that all tests
	// are being executed. If a test is added, this number must be changed.
	const expectedCompletionsCount = 64
	const expectedDiagnosticsCount = 16
	const expectedFormatCount = 4
	const expectedDefinitionsCount = 16
	const expectedTypeDefinitionsCount = 2
	const expectedHighlightsCount = 2
	const expectedSymbolsCount = 1

	files := packagestest.MustCopyFileTree(dir)
	for fragment, operation := range files {
		if trimmed := strings.TrimSuffix(fragment, ".in"); trimmed != fragment {
			delete(files, fragment)
			files[trimmed] = operation
		}
	}
	modules := []packagestest.Module{
		{
			Name:  "golang.org/x/tools/internal/lsp",
			Files: files,
		},
	}
	exported := packagestest.Export(t, exporter, modules)
	defer exported.Cleanup()

	// Merge the exported.Config with the view.Config.
	cfg := *exported.Config
	cfg.Fset = token.NewFileSet()
	cfg.Context = context.Background()
	cfg.ParseFile = func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
		return parser.ParseFile(fset, filename, src, parser.AllErrors|parser.ParseComments)
	}

	log := xlog.New(xlog.StdSink{})
	s := &Server{
		views:       []*cache.View{cache.NewView(ctx, log, "lsp_test", span.FileURI(cfg.Dir), &cfg)},
		undelivered: make(map[span.URI][]source.Diagnostic),
	}
	// Do a first pass to collect special markers for completion.
	if err := exported.Expect(map[string]interface{}{
		"item": func(name string, r packagestest.Range, _, _ string) {
			exported.Mark(name, r)
		},
	}); err != nil {
		t.Fatal(err)
	}

	expectedDiagnostics := make(diagnostics)
	completionItems := make(completionItems)
	expectedCompletions := make(completions)
	expectedFormat := make(formats)
	expectedDefinitions := make(definitions)
	expectedTypeDefinitions := make(definitions)
	expectedHighlights := make(highlights)
	expectedSymbols := make(symbols)

	// Collect any data that needs to be used by subsequent tests.
	if err := exported.Expect(map[string]interface{}{
		"diag":      expectedDiagnostics.collect,
		"item":      completionItems.collect,
		"complete":  expectedCompletions.collect,
		"format":    expectedFormat.collect,
		"godef":     expectedDefinitions.collect,
		"typdef":    expectedTypeDefinitions.collect,
		"highlight": expectedHighlights.collect,
		"symbol":    expectedSymbols.collect,
	}); err != nil {
		t.Fatal(err)
	}

	t.Run("Completion", func(t *testing.T) {
		t.Helper()
		if goVersion111 { // TODO(rstambler): Remove this when we no longer support Go 1.10.
			if len(expectedCompletions) != expectedCompletionsCount {
				t.Errorf("got %v completions expected %v", len(expectedCompletions), expectedCompletionsCount)
			}
		}
		expectedCompletions.test(t, exported, s, completionItems)
	})

	t.Run("Diagnostics", func(t *testing.T) {
		t.Helper()
		diagnosticsCount := expectedDiagnostics.test(t, s.views[0])
		if goVersion111 { // TODO(rstambler): Remove this when we no longer support Go 1.10.
			if diagnosticsCount != expectedDiagnosticsCount {
				t.Errorf("got %v diagnostics expected %v", diagnosticsCount, expectedDiagnosticsCount)
			}
		}
	})

	t.Run("Format", func(t *testing.T) {
		if _, err := exec.LookPath("gofmt"); err != nil {
			switch runtime.GOOS {
			case "android":
				t.Skip("gofmt is not installed")
			default:
				t.Fatal(err)
			}
		}
		t.Helper()
		if goVersion111 { // TODO(rstambler): Remove this when we no longer support Go 1.10.
			if len(expectedFormat) != expectedFormatCount {
				t.Errorf("got %v formats expected %v", len(expectedFormat), expectedFormatCount)
			}
		}
		expectedFormat.test(t, s)
	})

	t.Run("Definitions", func(t *testing.T) {
		t.Helper()
		if goVersion111 { // TODO(rstambler): Remove this when we no longer support Go 1.10.
			if len(expectedDefinitions) != expectedDefinitionsCount {
				t.Errorf("got %v definitions expected %v", len(expectedDefinitions), expectedDefinitionsCount)
			}
		}
		expectedDefinitions.test(t, s, false)
	})

	t.Run("TypeDefinitions", func(t *testing.T) {
		t.Helper()
		if goVersion111 { // TODO(rstambler): Remove this when we no longer support Go 1.10.
			if len(expectedTypeDefinitions) != expectedTypeDefinitionsCount {
				t.Errorf("got %v type definitions expected %v", len(expectedTypeDefinitions), expectedTypeDefinitionsCount)
			}
		}
		expectedTypeDefinitions.test(t, s, true)
	})

	t.Run("Highlights", func(t *testing.T) {
		t.Helper()
		if goVersion111 { // TODO(rstambler): Remove this when we no longer support Go 1.10.
			if len(expectedHighlights) != expectedHighlightsCount {
				t.Errorf("got %v highlights expected %v", len(expectedHighlights), expectedHighlightsCount)
			}
		}
		expectedHighlights.test(t, s)
	})

	t.Run("Symbols", func(t *testing.T) {
		t.Helper()
		if goVersion111 { // TODO(rstambler): Remove this when we no longer support Go 1.10.
			if len(expectedSymbols) != expectedSymbolsCount {
				t.Errorf("got %v symbols expected %v", len(expectedSymbols), expectedSymbolsCount)
			}
		}
		expectedSymbols.test(t, s)
	})
}

type diagnostics map[span.URI][]protocol.Diagnostic
type completionItems map[token.Pos]*protocol.CompletionItem
type completions map[token.Position][]token.Pos
type formats map[string]string
type definitions map[protocol.Location]protocol.Location
type highlights map[string][]protocol.Location
type symbols map[span.URI][]protocol.DocumentSymbol

func (d diagnostics) test(t *testing.T, v source.View) int {
	count := 0
	ctx := context.Background()
	for uri, want := range d {
		sourceDiagnostics, err := source.Diagnostics(context.Background(), v, uri)
		if err != nil {
			t.Fatal(err)
		}
		got, err := toProtocolDiagnostics(ctx, v, sourceDiagnostics[uri])
		if err != nil {
			t.Fatal(err)
		}
		sorted(got)
		if diff := diffDiagnostics(uri, want, got); diff != "" {
			t.Error(diff)
		}
		count += len(want)
	}
	return count
}

func (d diagnostics) collect(e *packagestest.Exported, fset *token.FileSet, rng packagestest.Range, msgSource, msg string) {
	spn, m := testLocation(e, fset, rng)
	if _, ok := d[spn.URI()]; !ok {
		d[spn.URI()] = []protocol.Diagnostic{}
	}
	// If a file has an empty diagnostic message, return. This allows us to
	// avoid testing diagnostics in files that may have a lot of them.
	if msg == "" {
		return
	}
	severity := protocol.SeverityError
	if strings.Contains(string(spn.URI()), "analyzer") {
		severity = protocol.SeverityWarning
	}
	dRng, err := m.Range(spn)
	if err != nil {
		return
	}
	want := protocol.Diagnostic{
		Range:    dRng,
		Severity: severity,
		Source:   msgSource,
		Message:  msg,
	}
	d[spn.URI()] = append(d[spn.URI()], want)
}

// diffDiagnostics prints the diff between expected and actual diagnostics test
// results.
func diffDiagnostics(uri span.URI, want, got []protocol.Diagnostic) string {
	if len(got) != len(want) {
		goto Failed
	}
	for i, w := range want {
		g := got[i]
		if w.Message != g.Message {
			goto Failed
		}
		if w.Range.Start != g.Range.Start {
			goto Failed
		}
		// Special case for diagnostics on parse errors.
		if strings.Contains(string(uri), "noparse") {
			if g.Range.Start != g.Range.End || w.Range.Start != g.Range.End {
				goto Failed
			}
		} else if g.Range.End != g.Range.Start { // Accept any 'want' range if the diagnostic returns a zero-length range.
			if w.Range.End != g.Range.End {
				goto Failed
			}
		}
		if w.Severity != g.Severity {
			goto Failed
		}
		if w.Source != g.Source {
			goto Failed
		}
	}
	return ""
Failed:
	msg := &bytes.Buffer{}
	fmt.Fprintf(msg, "diagnostics failed for %s:\nexpected:\n", uri)
	for _, d := range want {
		fmt.Fprintf(msg, "  %v\n", d)
	}
	fmt.Fprintf(msg, "got:\n")
	for _, d := range got {
		fmt.Fprintf(msg, "  %v\n", d)
	}
	return msg.String()
}

func (c completions) test(t *testing.T, exported *packagestest.Exported, s *Server, items completionItems) {
	for src, itemList := range c {
		var want []protocol.CompletionItem
		for _, pos := range itemList {
			want = append(want, *items[pos])
		}
		list, err := s.Completion(context.Background(), &protocol.CompletionParams{
			TextDocumentPositionParams: protocol.TextDocumentPositionParams{
				TextDocument: protocol.TextDocumentIdentifier{
					URI: protocol.NewURI(span.FileURI(src.Filename)),
				},
				Position: protocol.Position{
					Line:      float64(src.Line - 1),
					Character: float64(src.Column - 1),
				},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		wantBuiltins := strings.Contains(src.Filename, "builtins")
		var got []protocol.CompletionItem
		for _, item := range list.Items {
			if !wantBuiltins && isBuiltin(item) {
				continue
			}
			got = append(got, item)
		}
		if err != nil {
			t.Fatalf("completion failed for %s:%v:%v: %v", filepath.Base(src.Filename), src.Line, src.Column, err)
		}
		if diff := diffCompletionItems(t, src, want, got); diff != "" {
			t.Errorf(diff)
		}
	}
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

func (c completions) collect(src token.Position, expected []token.Pos) {
	c[src] = expected
}

func (i completionItems) collect(pos token.Pos, label, detail, kind string) {
	var k protocol.CompletionItemKind
	switch kind {
	case "struct":
		k = protocol.StructCompletion
	case "func":
		k = protocol.FunctionCompletion
	case "var":
		k = protocol.VariableCompletion
	case "type":
		k = protocol.TypeParameterCompletion
	case "field":
		k = protocol.FieldCompletion
	case "interface":
		k = protocol.InterfaceCompletion
	case "const":
		k = protocol.ConstantCompletion
	case "method":
		k = protocol.MethodCompletion
	case "package":
		k = protocol.ModuleCompletion
	}
	i[pos] = &protocol.CompletionItem{
		Label:  label,
		Detail: detail,
		Kind:   k,
	}
}

// diffCompletionItems prints the diff between expected and actual completion
// test results.
func diffCompletionItems(t *testing.T, pos token.Position, want, got []protocol.CompletionItem) string {
	if len(got) != len(want) {
		goto Failed
	}
	for i, w := range want {
		g := got[i]
		if w.Label != g.Label {
			goto Failed
		}
		if w.Detail != g.Detail {
			goto Failed
		}
		if w.Kind != g.Kind {
			goto Failed
		}
	}
	return ""
Failed:
	msg := &bytes.Buffer{}
	fmt.Fprintf(msg, "completion failed for %s:%v:%v:\nexpected:\n", filepath.Base(pos.Filename), pos.Line, pos.Column)
	for _, d := range want {
		fmt.Fprintf(msg, "  %v\n", d)
	}
	fmt.Fprintf(msg, "got:\n")
	for _, d := range got {
		fmt.Fprintf(msg, "  %v\n", d)
	}
	return msg.String()
}

func (f formats) test(t *testing.T, s *Server) {
	ctx := context.Background()
	for filename, gofmted := range f {
		uri := span.FileURI(filename)
		edits, err := s.Formatting(context.Background(), &protocol.DocumentFormattingParams{
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
		f, m, err := newColumnMap(ctx, s.findView(ctx, uri), uri)
		if err != nil {
			t.Error(err)
		}
		buf, err := applyEdits(m, f.GetContent(context.Background()), edits)
		if err != nil {
			t.Error(err)
		}
		got := string(buf)
		if gofmted != got {
			t.Errorf("format failed for %s: expected '%v', got '%v'", filename, gofmted, got)
		}
	}
}

func (f formats) collect(pos token.Position) {
	cmd := exec.Command("gofmt", pos.Filename)
	stdout := bytes.NewBuffer(nil)
	cmd.Stdout = stdout
	cmd.Run() // ignore error, sometimes we have intentionally ungofmt-able files
	f[pos.Filename] = stdout.String()
}

func (d definitions) test(t *testing.T, s *Server, typ bool) {
	for src, target := range d {
		params := &protocol.TextDocumentPositionParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: src.URI,
			},
			Position: src.Range.Start,
		}
		var locs []protocol.Location
		var err error
		if typ {
			locs, err = s.TypeDefinition(context.Background(), params)
		} else {
			locs, err = s.Definition(context.Background(), params)
		}
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}
		if len(locs) != 1 {
			t.Errorf("got %d locations for definition, expected 1", len(locs))
		}
		if locs[0] != target {
			t.Errorf("for %v got %v want %v", src, locs[0], target)
		}
	}
}

func (d definitions) collect(e *packagestest.Exported, fset *token.FileSet, src, target packagestest.Range) {
	sSrc, mSrc := testLocation(e, fset, src)
	lSrc, err := mSrc.Location(sSrc)
	if err != nil {
		return
	}
	sTarget, mTarget := testLocation(e, fset, target)
	lTarget, err := mTarget.Location(sTarget)
	if err != nil {
		return
	}
	d[lSrc] = lTarget
}

func (h highlights) collect(e *packagestest.Exported, fset *token.FileSet, name string, rng packagestest.Range) {
	s, m := testLocation(e, fset, rng)
	loc, err := m.Location(s)
	if err != nil {
		return
	}

	h[name] = append(h[name], loc)
}

func (h highlights) test(t *testing.T, s *Server) {
	for name, locations := range h {
		params := &protocol.TextDocumentPositionParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: locations[0].URI,
			},
			Position: locations[0].Range.Start,
		}
		highlights, err := s.DocumentHighlight(context.Background(), params)
		if err != nil {
			t.Fatal(err)
		}
		if len(highlights) != len(locations) {
			t.Fatalf("got %d highlights for %s, expected %d", len(highlights), name, len(locations))
		}
		for i := range highlights {
			if highlights[i].Range != locations[i].Range {
				t.Errorf("want %v, got %v\n", locations[i].Range, highlights[i].Range)
			}
		}
	}
}

func (s symbols) collect(e *packagestest.Exported, fset *token.FileSet, name string, rng span.Range, kind int64) {
	f := fset.File(rng.Start)
	if f == nil {
		return
	}

	content, err := e.FileContents(f.Name())
	if err != nil {
		return
	}

	spn, err := rng.Span()
	if err != nil {
		return
	}

	m := protocol.NewColumnMapper(spn.URI(), fset, f, content)
	prng, err := m.Range(spn)
	if err != nil {
		return
	}

	s[spn.URI()] = append(s[spn.URI()], protocol.DocumentSymbol{
		Name:           name,
		Kind:           protocol.SymbolKind(kind),
		SelectionRange: prng,
	})
}

func (s symbols) test(t *testing.T, server *Server) {
	for uri, expectedSymbols := range s {
		params := &protocol.DocumentSymbolParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: string(uri),
			},
		}
		symbols, err := server.DocumentSymbol(context.Background(), params)
		if err != nil {
			t.Fatal(err)
		}

		if len(symbols) != len(expectedSymbols) {
			t.Errorf("want %d symbols in %v, got %d", len(expectedSymbols), uri, len(symbols))
			continue
		}

		sort.Slice(symbols, func(i, j int) bool { return symbols[i].Name < symbols[j].Name })
		sort.Slice(expectedSymbols, func(i, j int) bool { return expectedSymbols[i].Name < expectedSymbols[j].Name })
		for i, w := range expectedSymbols {
			g := symbols[i]
			if w.Name != g.Name {
				t.Errorf("%s: want symbol %q, got %q", uri, w.Name, g.Name)
				continue
			}
			if w.Kind != g.Kind {
				t.Errorf("%s: want kind %v for %s, got %v", uri, w.Kind, w.Name, g.Kind)
			}
			if w.SelectionRange != g.SelectionRange {
				t.Errorf("%s: want selection range %v for %s, got %v", uri, w.SelectionRange, w.Name, g.SelectionRange)
			}
		}
	}
}

func testLocation(e *packagestest.Exported, fset *token.FileSet, rng packagestest.Range) (span.Span, *protocol.ColumnMapper) {
	spn, err := span.NewRange(fset, rng.Start, rng.End).Span()
	if err != nil {
		return spn, nil
	}
	f := fset.File(rng.Start)
	content, err := e.FileContents(f.Name())
	if err != nil {
		return spn, nil
	}
	m := protocol.NewColumnMapper(spn.URI(), fset, f, content)
	return spn, m
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
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 0, Character: 4}, want: -1},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 1, Character: 0}, want: 4},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 1, Character: 3}, want: 7},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 1, Character: 4}, want: -1},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 2, Character: 0}, want: 8},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 2, Character: 1}, want: -1},
		{text: "aaa\nbbb\n\n", pos: protocol.Position{Line: 2, Character: 0}, want: 8},
	}

	for i, test := range tests {
		fname := fmt.Sprintf("test %d", i)
		fset := token.NewFileSet()
		f := fset.AddFile(fname, -1, len(test.text))
		f.SetLinesForContent([]byte(test.text))
		mapper := protocol.NewColumnMapper(span.FileURI(fname), fset, f, []byte(test.text))
		got, err := mapper.Point(test.pos)
		if err != nil && test.want != -1 {
			t.Errorf("unexpected error: %v", err)
		}
		if err == nil && got.Offset() != test.want {
			t.Errorf("want %d for %q(Line:%d,Character:%d), but got %d", test.want, test.text, int(test.pos.Line), int(test.pos.Character), got.Offset())
		}
	}
}

func applyEdits(m *protocol.ColumnMapper, content []byte, edits []protocol.TextEdit) ([]byte, error) {
	prev := 0
	result := make([]byte, 0, len(content))
	for _, edit := range edits {
		spn, err := m.RangeSpan(edit.Range)
		if err != nil {
			return nil, err
		}
		offset := spn.Start().Offset()
		if offset > prev {
			result = append(result, content[prev:offset]...)
		}
		if len(edit.NewText) > 0 {
			result = append(result, []byte(edit.NewText)...)
		}
		prev = spn.End().Offset()
	}
	if prev < len(content) {
		result = append(result, content[prev:]...)
	}
	return result, nil
}
