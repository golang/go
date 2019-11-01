// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"fmt"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	testenv.ExitIfSmallMachine()
	os.Exit(m.Run())
}

func TestLSP(t *testing.T) {
	packagestest.TestAll(t, testLSP)
}

type runner struct {
	server *Server
	data   *tests.Data
	ctx    context.Context
}

const viewName = "lsp_test"

func testLSP(t *testing.T, exporter packagestest.Exporter) {
	ctx := tests.Context(t)
	data := tests.Load(t, exporter, "testdata")
	defer data.Exported.Cleanup()

	cache := cache.New(nil)
	session := cache.NewSession(ctx)
	options := tests.DefaultOptions()
	session.SetOptions(options)
	options.Env = data.Config.Env
	_, err := session.NewView(ctx, viewName, span.FileURI(data.Config.Dir), options)
	if err != nil {
		t.Fatal(err)
	}
	for filename, content := range data.Config.Overlay {
		session.SetOverlay(span.FileURI(filename), source.DetectLanguage("", filename), content)
	}
	r := &runner{
		server: &Server{
			session:     session,
			undelivered: make(map[span.URI][]source.Diagnostic),
		},
		data: data,
		ctx:  ctx,
	}

	tests.Run(t, r, data)
}

// TODO: Actually test the LSP diagnostics function in this test.
func (r *runner) Diagnostics(t *testing.T, uri span.URI, want []source.Diagnostic) {
	v := r.server.session.View(viewName)
	f, err := v.GetFile(r.ctx, uri)
	if err != nil {
		t.Fatalf("no file for %s: %v", f, err)
	}
	results, _, err := source.Diagnostics(r.ctx, v, f, nil)
	if err != nil {
		t.Fatal(err)
	}
	got := results[uri]
	// A special case to test that there are no diagnostics for a file.
	if len(want) == 1 && want[0].Source == "no_diagnostics" {
		if len(got) != 0 {
			t.Errorf("expected no diagnostics for %s, got %v", uri, got)
		}
		return
	}
	if diff := tests.DiffDiagnostics(want, got); diff != "" {
		t.Error(diff)
	}
}

func (r *runner) FoldingRange(t *testing.T, spn span.Span) {
	uri := spn.URI()
	view := r.server.session.ViewOf(uri)
	original := view.Options()
	modified := original

	// Test all folding ranges.
	modified.LineFoldingOnly = false
	view.SetOptions(modified)
	ranges, err := r.server.FoldingRange(r.ctx, &protocol.FoldingRangeParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.NewURI(uri),
		},
	})
	if err != nil {
		t.Error(err)
		return
	}
	r.foldingRanges(t, "foldingRange", uri, ranges)

	// Test folding ranges with lineFoldingOnly = true.
	modified.LineFoldingOnly = true
	view.SetOptions(modified)
	ranges, err = r.server.FoldingRange(r.ctx, &protocol.FoldingRangeParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.NewURI(uri),
		},
	})
	if err != nil {
		t.Error(err)
		return
	}
	r.foldingRanges(t, "foldingRange-lineFolding", uri, ranges)
	view.SetOptions(original)
}

func (r *runner) foldingRanges(t *testing.T, prefix string, uri span.URI, ranges []protocol.FoldingRange) {
	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	// Fold all ranges.
	nonOverlapping := nonOverlappingRanges(ranges)
	for i, rngs := range nonOverlapping {
		got, err := foldRanges(m, string(m.Content), rngs)
		if err != nil {
			t.Error(err)
			continue
		}
		tag := fmt.Sprintf("%s-%d", prefix, i)
		want := string(r.data.Golden(tag, uri.Filename(), func() ([]byte, error) {
			return []byte(got), nil
		}))

		if want != got {
			t.Errorf("%s: foldingRanges failed for %s, expected:\n%v\ngot:\n%v", tag, uri.Filename(), want, got)
		}
	}

	// Filter by kind.
	kinds := []protocol.FoldingRangeKind{protocol.Imports, protocol.Comment}
	for _, kind := range kinds {
		var kindOnly []protocol.FoldingRange
		for _, fRng := range ranges {
			if fRng.Kind == string(kind) {
				kindOnly = append(kindOnly, fRng)
			}
		}

		nonOverlapping := nonOverlappingRanges(kindOnly)
		for i, rngs := range nonOverlapping {
			got, err := foldRanges(m, string(m.Content), rngs)
			if err != nil {
				t.Error(err)
				continue
			}
			tag := fmt.Sprintf("%s-%s-%d", prefix, kind, i)
			want := string(r.data.Golden(tag, uri.Filename(), func() ([]byte, error) {
				return []byte(got), nil
			}))

			if want != got {
				t.Errorf("%s: foldingRanges failed for %s, expected:\n%v\ngot:\n%v", tag, uri.Filename(), want, got)
			}
		}

	}
}

func nonOverlappingRanges(ranges []protocol.FoldingRange) (res [][]protocol.FoldingRange) {
	for _, fRng := range ranges {
		setNum := len(res)
		for i := 0; i < len(res); i++ {
			canInsert := true
			for _, rng := range res[i] {
				if conflict(rng, fRng) {
					canInsert = false
					break
				}
			}
			if canInsert {
				setNum = i
				break
			}
		}
		if setNum == len(res) {
			res = append(res, []protocol.FoldingRange{})
		}
		res[setNum] = append(res[setNum], fRng)
	}
	return res
}

func conflict(a, b protocol.FoldingRange) bool {
	// a start position is <= b start positions
	return (a.StartLine < b.StartLine || (a.StartLine == b.StartLine && a.StartCharacter <= b.StartCharacter)) &&
		(a.EndLine > b.StartLine || (a.EndLine == b.StartLine && a.EndCharacter > b.StartCharacter))
}

func foldRanges(m *protocol.ColumnMapper, contents string, ranges []protocol.FoldingRange) (string, error) {
	foldedText := "<>"
	res := contents
	// Apply the edits from the end of the file forward
	// to preserve the offsets
	for i := len(ranges) - 1; i >= 0; i-- {
		fRange := ranges[i]
		spn, err := m.RangeSpan(protocol.Range{
			Start: protocol.Position{
				Line:      fRange.StartLine,
				Character: fRange.StartCharacter,
			},
			End: protocol.Position{
				Line:      fRange.EndLine,
				Character: fRange.EndCharacter,
			},
		})
		if err != nil {
			return "", err
		}
		start := spn.Start().Offset()
		end := spn.End().Offset()

		tmp := res[0:start] + foldedText
		res = tmp + res[end:]
	}
	return res, nil
}

func (r *runner) Format(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()
	gofmted := string(r.data.Golden("gofmt", filename, func() ([]byte, error) {
		cmd := exec.Command("gofmt", filename)
		out, _ := cmd.Output() // ignore error, sometimes we have intentionally ungofmt-able files
		return out, nil
	}))

	edits, err := r.server.Formatting(r.ctx, &protocol.DocumentFormattingParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.NewURI(uri),
		},
	})
	if err != nil {
		if gofmted != "" {
			t.Error(err)
		}
		return
	}
	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	sedits, err := source.FromProtocolEdits(m, edits)
	if err != nil {
		t.Error(err)
	}
	got := diff.ApplyEdits(string(m.Content), sedits)
	if gofmted != got {
		t.Errorf("format failed for %s, expected:\n%v\ngot:\n%v", filename, gofmted, got)
	}
}

func (r *runner) Import(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()
	goimported := string(r.data.Golden("goimports", filename, func() ([]byte, error) {
		cmd := exec.Command("goimports", filename)
		out, _ := cmd.Output() // ignore error, sometimes we have intentionally ungofmt-able files
		return out, nil
	}))

	actions, err := r.server.CodeAction(r.ctx, &protocol.CodeActionParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.NewURI(uri),
		},
	})
	if err != nil {
		if goimported != "" {
			t.Error(err)
		}
		return
	}
	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	var edits []protocol.TextEdit
	for _, a := range actions {
		if a.Title == "Organize Imports" {
			edits = (*a.Edit.Changes)[string(uri)]
		}
	}
	sedits, err := source.FromProtocolEdits(m, edits)
	if err != nil {
		t.Error(err)
	}
	got := diff.ApplyEdits(string(m.Content), sedits)
	if goimported != got {
		t.Errorf("import failed for %s, expected:\n%v\ngot:\n%v", filename, goimported, got)
	}
}

func (r *runner) SuggestedFix(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()
	view := r.server.session.ViewOf(uri)
	f, err := view.GetFile(r.ctx, uri)
	if err != nil {
		t.Fatal(err)
	}
	diagnostics, _, err := source.Diagnostics(r.ctx, view, f, nil)
	if err != nil {
		t.Fatal(err)
	}
	actions, err := r.server.CodeAction(r.ctx, &protocol.CodeActionParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.NewURI(uri),
		},
		Context: protocol.CodeActionContext{
			Only:        []protocol.CodeActionKind{protocol.QuickFix},
			Diagnostics: toProtocolDiagnostics(r.ctx, diagnostics[uri]),
		},
	})
	if err != nil {
		t.Error(err)
		return
	}
	m, err := r.data.Mapper(f.URI())
	if err != nil {
		t.Fatal(err)
	}
	var edits []protocol.TextEdit
	for _, a := range actions {
		if a.Title == "Remove" {
			edits = (*a.Edit.Changes)[string(uri)]
		}
	}
	sedits, err := source.FromProtocolEdits(m, edits)
	if err != nil {
		t.Error(err)
	}
	got := diff.ApplyEdits(string(m.Content), sedits)
	fixed := string(r.data.Golden("suggestedfix", filename, func() ([]byte, error) {
		return []byte(got), nil
	}))
	if fixed != got {
		t.Errorf("suggested fixes failed for %s, expected:\n%v\ngot:\n%v", filename, fixed, got)
	}
}

func (r *runner) Definition(t *testing.T, spn span.Span, d tests.Definition) {
	sm, err := r.data.Mapper(d.Src.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := sm.Location(d.Src)
	if err != nil {
		t.Fatalf("failed for %v: %v", d.Src, err)
	}
	tdpp := protocol.TextDocumentPositionParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
	}
	var locs []protocol.Location
	var hover *protocol.Hover
	if d.IsType {
		params := &protocol.TypeDefinitionParams{
			TextDocumentPositionParams: tdpp,
		}
		locs, err = r.server.TypeDefinition(r.ctx, params)
	} else {
		params := &protocol.DefinitionParams{
			TextDocumentPositionParams: tdpp,
		}
		locs, err = r.server.Definition(r.ctx, params)
		if err != nil {
			t.Fatalf("failed for %v: %+v", d.Src, err)
		}
		v := &protocol.HoverParams{
			TextDocumentPositionParams: tdpp,
		}
		hover, err = r.server.Hover(r.ctx, v)
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
		lm, err := r.data.Mapper(locURI)
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

func (r *runner) Implementation(t *testing.T, spn span.Span, m tests.Implementations) {
	sm, err := r.data.Mapper(m.Src.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := sm.Location(m.Src)
	if err != nil {
		t.Fatalf("failed for %v: %v", m.Src, err)
	}
	tdpp := protocol.TextDocumentPositionParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
	}
	var locs []protocol.Location
	params := &protocol.ImplementationParams{
		TextDocumentPositionParams: tdpp,
	}
	locs, err = r.server.Implementation(r.ctx, params)
	if err != nil {
		t.Fatalf("failed for %v: %v", m.Src, err)
	}
	if len(locs) != len(m.Implementations) {
		t.Fatalf("got %d locations for implementation, expected %d", len(locs), len(m.Implementations))
	}
	for i := range locs {
		locURI := span.NewURI(locs[i].URI)
		lm, err := r.data.Mapper(locURI)
		if err != nil {
			t.Fatal(err)
		}
		if imp, err := lm.Span(locs[i]); err != nil {
			t.Fatalf("failed for %v: %v", locs[i], err)
		} else if imp != m.Implementations[i] {
			t.Errorf("for %dth implementation of %v got %v want %v", i, m.Src, imp, m.Implementations[i])
		}
	}
}

func (r *runner) Highlight(t *testing.T, name string, locations []span.Span) {
	m, err := r.data.Mapper(locations[0].URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := m.Location(locations[0])
	if err != nil {
		t.Fatalf("failed for %v: %v", locations[0], err)
	}
	tdpp := protocol.TextDocumentPositionParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
	}
	params := &protocol.DocumentHighlightParams{
		TextDocumentPositionParams: tdpp,
	}
	highlights, err := r.server.DocumentHighlight(r.ctx, params)
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

func (r *runner) References(t *testing.T, src span.Span, itemList []span.Span) {
	sm, err := r.data.Mapper(src.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := sm.Location(src)
	if err != nil {
		t.Fatalf("failed for %v: %v", src, err)
	}

	want := make(map[protocol.Location]bool)
	for _, pos := range itemList {
		m, err := r.data.Mapper(pos.URI())
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
	got, err := r.server.References(r.ctx, params)
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

func (r *runner) Rename(t *testing.T, spn span.Span, newText string) {
	tag := fmt.Sprintf("%s-rename", newText)

	uri := spn.URI()
	filename := uri.Filename()
	sm, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	loc, err := sm.Location(spn)
	if err != nil {
		t.Fatalf("failed for %v: %v", spn, err)
	}

	workspaceEdits, err := r.server.Rename(r.ctx, &protocol.RenameParams{
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
		return
	}

	var res []string
	for uri, edits := range *workspaceEdits.Changes {
		m, err := r.data.Mapper(span.URI(uri))
		if err != nil {
			t.Fatal(err)
		}
		sedits, err := source.FromProtocolEdits(m, edits)
		if err != nil {
			t.Error(err)
		}
		filename := filepath.Base(m.URI.Filename())
		contents := applyEdits(string(m.Content), sedits)
		if len(*workspaceEdits.Changes) > 1 {
			contents = fmt.Sprintf("%s:\n%s", filename, contents)
		}
		res = append(res, contents)
	}

	// Sort on filename
	sort.Strings(res)

	var got string
	for i, val := range res {
		if i != 0 {
			got += "\n"
		}
		got += val
	}

	renamed := string(r.data.Golden(tag, filename, func() ([]byte, error) {
		return []byte(got), nil
	}))

	if renamed != got {
		t.Errorf("rename failed for %s, expected:\n%v\ngot:\n%v", newText, renamed, got)
	}
}

func (r *runner) PrepareRename(t *testing.T, src span.Span, want *source.PrepareItem) {
	m, err := r.data.Mapper(src.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := m.Location(src)
	if err != nil {
		t.Fatalf("failed for %v: %v", src, err)
	}
	tdpp := protocol.TextDocumentPositionParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
	}
	params := &protocol.PrepareRenameParams{
		TextDocumentPositionParams: tdpp,
	}
	got, err := r.server.PrepareRename(context.Background(), params)
	if err != nil {
		t.Errorf("prepare rename failed for %v: got error: %v", src, err)
		return
	}
	if got == nil {
		if want.Text != "" { // expected an ident.
			t.Errorf("prepare rename failed for %v: got nil", src)
		}
		return
	}
	if protocol.CompareRange(*got, want.Range) != 0 {
		t.Errorf("prepare rename failed: incorrect range got %v want %v", *got, want.Range)
	}
}

func applyEdits(contents string, edits []diff.TextEdit) string {
	res := contents

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

func (r *runner) Symbols(t *testing.T, uri span.URI, expectedSymbols []protocol.DocumentSymbol) {
	params := &protocol.DocumentSymbolParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: string(uri),
		},
	}
	symbols, err := r.server.DocumentSymbol(r.ctx, params)
	if err != nil {
		t.Fatal(err)
	}

	if len(symbols) != len(expectedSymbols) {
		t.Errorf("want %d top-level symbols in %v, got %d", len(expectedSymbols), uri, len(symbols))
		return
	}
	if diff := r.diffSymbols(t, uri, expectedSymbols, symbols); diff != "" {
		t.Error(diff)
	}
}

func (r *runner) diffSymbols(t *testing.T, uri span.URI, want []protocol.DocumentSymbol, got []protocol.DocumentSymbol) string {
	sort.Slice(want, func(i, j int) bool { return want[i].Name < want[j].Name })
	sort.Slice(got, func(i, j int) bool { return got[i].Name < got[j].Name })
	if len(got) != len(want) {
		return summarizeSymbols(t, -1, want, got, "different lengths got %v want %v", len(got), len(want))
	}
	for i, w := range want {
		g := got[i]
		if w.Name != g.Name {
			return summarizeSymbols(t, i, want, got, "incorrect name got %v want %v", g.Name, w.Name)
		}
		if w.Kind != g.Kind {
			return summarizeSymbols(t, i, want, got, "incorrect kind got %v want %v", g.Kind, w.Kind)
		}
		if protocol.CompareRange(g.SelectionRange, w.SelectionRange) != 0 {
			return summarizeSymbols(t, i, want, got, "incorrect span got %v want %v", g.SelectionRange, w.SelectionRange)
		}
		if msg := r.diffSymbols(t, uri, w.Children, g.Children); msg != "" {
			return fmt.Sprintf("children of %s: %s", w.Name, msg)
		}
	}
	return ""
}

func summarizeSymbols(t *testing.T, i int, want, got []protocol.DocumentSymbol, reason string, args ...interface{}) string {
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

func (r *runner) SignatureHelp(t *testing.T, spn span.Span, expectedSignature *source.SignatureInformation) {
	m, err := r.data.Mapper(spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := m.Location(spn)
	if err != nil {
		t.Fatalf("failed for %v: %v", loc, err)
	}
	tdpp := protocol.TextDocumentPositionParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.NewURI(spn.URI()),
		},
		Position: loc.Range.Start,
	}
	params := &protocol.SignatureHelpParams{
		TextDocumentPositionParams: tdpp,
	}
	gotSignatures, err := r.server.SignatureHelp(r.ctx, params)
	if err != nil {
		// Only fail if we got an error we did not expect.
		if expectedSignature != nil {
			t.Fatal(err)
		}
		return
	}
	if expectedSignature == nil {
		if gotSignatures != nil {
			t.Errorf("expected no signature, got %v", gotSignatures)
		}
		return
	}
	if gotSignatures == nil {
		t.Fatalf("expected %v, got nil", expectedSignature)
	}
	if diff := diffSignatures(spn, expectedSignature, gotSignatures); diff != "" {
		t.Error(diff)
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

func (r *runner) Link(t *testing.T, uri span.URI, wantLinks []tests.Link) {
	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	got, err := r.server.DocumentLink(r.ctx, &protocol.DocumentLinkParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.NewURI(uri),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if diff := tests.DiffLinks(m, wantLinks, got); diff != "" {
		t.Error(diff)
	}
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
		uri := span.FileURI(fname)
		converter := span.NewContentConverter(fname, []byte(test.text))
		mapper := &protocol.ColumnMapper{
			URI:       uri,
			Converter: converter,
			Content:   []byte(test.text),
		}
		got, err := mapper.Point(test.pos)
		if err != nil && test.want != -1 {
			t.Errorf("unexpected error: %v", err)
		}
		if err == nil && got.Offset() != test.want {
			t.Errorf("want %d for %q(Line:%d,Character:%d), but got %d", test.want, test.text, int(test.pos.Line), int(test.pos.Character), got.Offset())
		}
	}
}
