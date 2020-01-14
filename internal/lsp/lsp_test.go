// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"fmt"
	"go/token"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/mod"
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
	session := cache.NewSession()
	options := tests.DefaultOptions()
	session.SetOptions(options)
	options.Env = data.Config.Env
	if _, _, err := session.NewView(ctx, viewName, span.FileURI(data.Config.Dir), options); err != nil {
		t.Fatal(err)
	}
	var modifications []source.FileModification
	for filename, content := range data.Config.Overlay {
		kind := source.DetectLanguage("", filename)
		if kind != source.Go {
			continue
		}
		modifications = append(modifications, source.FileModification{
			URI:        span.FileURI(filename),
			Action:     source.Open,
			Version:    -1,
			Text:       content,
			LanguageID: "go",
		})
	}
	if _, err := session.DidModifyFiles(ctx, modifications); err != nil {
		t.Fatal(err)
	}
	r := &runner{
		server: &Server{
			session:   session,
			delivered: map[span.URI]sentDiagnostics{},
		},
		data: data,
		ctx:  ctx,
	}
	tests.Run(t, r, data)
}

// TODO: Actually test the LSP diagnostics function in this test.
func (r *runner) Diagnostics(t *testing.T, uri span.URI, want []source.Diagnostic) {
	v := r.server.session.View(viewName)
	_, got, err := source.FileDiagnostics(r.ctx, v.Snapshot(), uri)
	if err != nil {
		t.Fatal(err)
	}
	// A special case to test that there are no diagnostics for a file.
	if len(want) == 1 && want[0].Source == "no_diagnostics" {
		if len(got) != 0 {
			t.Errorf("expected no diagnostics for %s, got %v", uri, got)
		}
		return
	}
	if diff := tests.DiffDiagnostics(uri, want, got); diff != "" {
		t.Error(diff)
	}
}

func (r *runner) FoldingRanges(t *testing.T, spn span.Span) {
	uri := spn.URI()
	view, err := r.server.session.ViewOf(uri)
	if err != nil {
		t.Fatal(err)
	}
	original := view.Options()
	modified := original

	// Test all folding ranges.
	modified.LineFoldingOnly = false
	view, err = view.SetOptions(r.ctx, modified)
	if err != nil {
		t.Error(err)
		return
	}
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
	view, err = view.SetOptions(r.ctx, modified)
	if err != nil {
		t.Error(err)
		return
	}
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
	view.SetOptions(r.ctx, original)
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
	actions, err := r.server.CodeAction(r.ctx, &protocol.CodeActionParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.NewURI(uri),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	got := string(m.Content)
	if len(actions) > 0 {
		res, err := applyWorkspaceEdits(r, actions[0].Edit)
		if err != nil {
			t.Fatal(err)
		}
		got = res[uri]
	}
	want := string(r.data.Golden("goimports", filename, func() ([]byte, error) {
		return []byte(got), nil
	}))
	if want != got {
		t.Errorf("import failed for %s, expected:\n%v\ngot:\n%v", filename, want, got)
	}
}

func (r *runner) SuggestedFix(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()
	view, err := r.server.session.ViewOf(uri)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := view.Snapshot()
	_, diagnostics, err := source.FileDiagnostics(r.ctx, snapshot, uri)
	if err != nil {
		t.Fatal(err)
	}
	actions, err := r.server.CodeAction(r.ctx, &protocol.CodeActionParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.NewURI(uri),
		},
		Context: protocol.CodeActionContext{
			Only:        []protocol.CodeActionKind{protocol.QuickFix},
			Diagnostics: toProtocolDiagnostics(diagnostics),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// TODO: This test should probably be able to handle multiple code actions.
	if len(actions) == 0 {
		t.Fatal("no code actions returned")
	}
	if len(actions) > 1 {
		t.Fatal("expected only 1 code action")
	}
	res, err := applyWorkspaceEdits(r, actions[0].Edit)
	if err != nil {
		t.Fatal(err)
	}
	got := res[uri]
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
	didSomething := false
	if hover != nil {
		didSomething = true
		tag := fmt.Sprintf("%s-hover", d.Name)
		expectHover := string(r.data.Golden(tag, d.Src.URI().Filename(), func() ([]byte, error) {
			return []byte(hover.Contents.Value), nil
		}))
		if hover.Contents.Value != expectHover {
			t.Errorf("for %v got %q want %q", d.Src, hover.Contents.Value, expectHover)
		}
	}
	if !d.OnlyHover {
		didSomething = true
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
	}
	if !didSomething {
		t.Errorf("no tests ran for %s", d.Src.URI())
	}
}

func (r *runner) Implementation(t *testing.T, spn span.Span, impls []span.Span) {
	sm, err := r.data.Mapper(spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := sm.Location(spn)
	if err != nil {
		t.Fatalf("failed for %v: %v", spn, err)
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
		t.Fatalf("failed for %v: %v", spn, err)
	}
	if len(locs) != len(impls) {
		t.Fatalf("got %d locations for implementation, expected %d", len(locs), len(impls))
	}

	var results []span.Span
	for i := range locs {
		locURI := span.NewURI(locs[i].URI)
		lm, err := r.data.Mapper(locURI)
		if err != nil {
			t.Fatal(err)
		}
		imp, err := lm.Span(locs[i])
		if err != nil {
			t.Fatalf("failed for %v: %v", locs[i], err)
		}
		results = append(results, imp)
	}
	// Sort results and expected to make tests deterministic.
	sort.SliceStable(results, func(i, j int) bool {
		return span.Compare(results[i], results[j]) == -1
	})
	sort.SliceStable(impls, func(i, j int) bool {
		return span.Compare(impls[i], impls[j]) == -1
	})
	for i := range results {
		if results[i] != impls[i] {
			t.Errorf("for %dth implementation of %v got %v want %v", i, spn, results[i], impls[i])
		}
	}
}

func (r *runner) Highlight(t *testing.T, src span.Span, locations []span.Span) {
	m, err := r.data.Mapper(src.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := m.Location(src)
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
		t.Fatalf("got %d highlights for highlight at %v:%v:%v, expected %d", len(highlights), src.URI().Filename(), src.Start().Line(), src.Start().Column(), len(locations))
	}
	// Check to make sure highlights have a valid range.
	var results []span.Span
	for i := range highlights {
		h, err := m.RangeSpan(highlights[i].Range)
		if err != nil {
			t.Fatalf("failed for %v: %v", highlights[i], err)
		}
		results = append(results, h)
	}
	// Sort results to make tests deterministic since DocumentHighlight uses a map.
	sort.SliceStable(results, func(i, j int) bool {
		return span.Compare(results[i], results[j]) == -1
	})
	// Check to make sure all the expected highlights are found.
	for i := range results {
		if results[i] != locations[i] {
			t.Errorf("want %v, got %v\n", locations[i], results[i])
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
	for _, includeDeclaration := range []bool{true, false} {
		t.Run(fmt.Sprintf("refs-declaration-%v", includeDeclaration), func(t *testing.T) {
			want := make(map[protocol.Location]bool)
			for i, pos := range itemList {
				// We don't want the first result if we aren't including the declaration.
				if i == 0 && !includeDeclaration {
					continue
				}
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
				Context: protocol.ReferenceContext{
					IncludeDeclaration: includeDeclaration,
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
		})

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

	wedit, err := r.server.Rename(r.ctx, &protocol.RenameParams{
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
	res, err := applyWorkspaceEdits(r, *wedit)
	if err != nil {
		t.Fatal(err)
	}
	var orderedURIs []string
	for uri := range res {
		orderedURIs = append(orderedURIs, string(uri))
	}
	sort.Strings(orderedURIs)

	var got string
	for i := 0; i < len(res); i++ {
		if i != 0 {
			got += "\n"
		}
		uri := span.URI(orderedURIs[i])
		if len(res) > 1 {
			got += filepath.Base(uri.Filename()) + ":\n"
		}
		val := res[uri]
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
	// we all love typed nils
	if got == nil || got.(*protocol.Range) == nil {
		if want.Text != "" { // expected an ident.
			t.Errorf("prepare rename failed for %v: got nil", src)
		}
		return
	}
	xx, ok := got.(*protocol.Range)
	if !ok {
		t.Fatalf("got %T, wanted Range", got)
	}
	if xx.Start == xx.End {
		// Special case for 0-length ranges. Marks can't specify a 0-length range,
		// so just compare the start.
		if xx.Start != want.Range.Start {
			t.Errorf("prepare rename failed: incorrect point, got %v want %v", xx.Start, want.Range.Start)
		}
	} else {
		if protocol.CompareRange(*xx, want.Range) != 0 {
			t.Errorf("prepare rename failed: incorrect range got %v want %v", *xx, want.Range)
		}
	}
}

func applyWorkspaceEdits(r *runner, wedit protocol.WorkspaceEdit) (map[span.URI]string, error) {
	res := map[span.URI]string{}
	for _, docEdits := range wedit.DocumentChanges {
		uri := span.URI(docEdits.TextDocument.URI)
		m, err := r.data.Mapper(uri)
		if err != nil {
			return nil, err
		}
		res[uri] = string(m.Content)
		sedits, err := source.FromProtocolEdits(m, docEdits.Edits)
		if err != nil {
			return nil, err
		}
		res[uri] = applyEdits(res[uri], sedits)
	}
	return res, nil
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
		return decorate("wanted active signature of 0, got %d", got.ActiveSignature)
	}

	if want.ActiveParameter != int(got.ActiveParameter) {
		return decorate("wanted active parameter of %d, got %d", want.ActiveParameter, got.ActiveParameter)
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

// TODO(golang/go#36091): This function can be refactored to look like the rest of this file
// when marker support gets added for go.mod files.
func TestModfileSuggestedFixes(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skip("this test cannot find mod/testdata files")
	}

	ctx := tests.Context(t)
	cache := cache.New(nil)
	session := cache.NewSession()
	options := tests.DefaultOptions()
	options.TempModfile = true
	options.Env = append(os.Environ(), "GOPACKAGESDRIVER=off", "GOROOT=")

	server := Server{
		session:   session,
		delivered: map[span.URI]sentDiagnostics{},
	}

	for _, tt := range []string{"indirect", "unused"} {
		t.Run(tt, func(t *testing.T) {
			folder, err := tests.CopyFolderToTempDir(filepath.Join("mod", "testdata", tt))
			if err != nil {
				t.Fatal(err)
			}
			defer os.RemoveAll(folder)

			_, snapshot, err := session.NewView(ctx, "suggested_fix_test", span.FileURI(folder), options)
			if err != nil {
				t.Fatal(err)
			}

			realURI, tempURI := snapshot.View().ModFiles()
			// TODO: Add testing for when the -modfile flag is turned off and we still get diagnostics.
			if tempURI == "" {
				return
			}
			realfh, err := snapshot.GetFile(realURI)
			if err != nil {
				t.Fatal(err)
			}

			reports, err := mod.Diagnostics(ctx, snapshot)
			if err != nil {
				t.Fatal(err)
			}
			if len(reports) != 1 {
				t.Errorf("expected 1 fileHandle, got %d", len(reports))
			}

			_, m, _, _, err := snapshot.ModTidyHandle(ctx, realfh).Tidy(ctx)
			if err != nil {
				t.Fatal(err)
			}

			for fh, diags := range reports {
				actions, err := server.CodeAction(ctx, &protocol.CodeActionParams{
					TextDocument: protocol.TextDocumentIdentifier{
						URI: protocol.NewURI(fh.URI),
					},
					Context: protocol.CodeActionContext{
						Only:        []protocol.CodeActionKind{protocol.SourceOrganizeImports},
						Diagnostics: toProtocolDiagnostics(diags),
					},
				})
				if err != nil {
					t.Fatal(err)
				}
				if len(actions) == 0 {
					t.Fatal("no code actions returned")
				}
				if len(actions) > 1 {
					t.Fatal("expected only 1 code action")
				}
				res := map[span.URI]string{}
				for _, docEdits := range actions[0].Edit.DocumentChanges {
					uri := span.URI(docEdits.TextDocument.URI)
					content, err := ioutil.ReadFile(uri.Filename())
					if err != nil {
						t.Fatal(err)
					}
					res[uri] = string(content)
					sedits, err := source.FromProtocolEdits(m, docEdits.Edits)
					if err != nil {
						t.Fatal(err)
					}
					res[uri] = applyEdits(res[uri], sedits)
				}
				got := res[realfh.Identity().URI]
				contents, err := ioutil.ReadFile(filepath.Join(folder, "go.mod.golden"))
				if err != nil {
					t.Fatal(err)
				}
				want := string(contents)
				if want != got {
					t.Errorf("suggested fixes failed for %s, expected:\n%s\ngot:\n%s", fh.URI.Filename(), want, got)
				}
			}
		})
	}
}
