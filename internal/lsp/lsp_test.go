// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
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
	"golang.org/x/tools/internal/lsp/diff/myers"
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
	server      *Server
	data        *tests.Data
	diagnostics map[span.URI][]*source.Diagnostic
	ctx         context.Context
}

func testLSP(t *testing.T, exporter packagestest.Exporter) {
	ctx := tests.Context(t)
	data := tests.Load(t, exporter, "testdata")

	for _, datum := range data {
		defer datum.Exported.Cleanup()

		cache := cache.New(ctx, nil)
		session := cache.NewSession(ctx)
		options := tests.DefaultOptions()
		session.SetOptions(options)
		options.Env = datum.Config.Env
		v, snapshot, err := session.NewView(ctx, datum.Config.Dir, span.URIFromPath(datum.Config.Dir), options)
		if err != nil {
			t.Fatal(err)
		}

		// Enable type error analyses for tests.
		// TODO(golang/go#38212): Delete this once they are enabled by default.
		tests.EnableAllAnalyzers(snapshot, &options)
		v.SetOptions(ctx, options)

		// Check to see if the -modfile flag is available, this is basically a check
		// to see if the go version >= 1.14. Otherwise, the modfile specific tests
		// will always fail if this flag is not available.
		for _, flag := range v.Snapshot().Config(ctx).BuildFlags {
			if strings.Contains(flag, "-modfile=") {
				datum.ModfileFlagAvailable = true
				break
			}
		}
		var modifications []source.FileModification
		for filename, content := range datum.Config.Overlay {
			kind := source.DetectLanguage("", filename)
			if kind != source.Go {
				continue
			}
			modifications = append(modifications, source.FileModification{
				URI:        span.URIFromPath(filename),
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
			server: NewServer(session, nil),
			data:   datum,
			ctx:    ctx,
		}
		t.Run(tests.FormatFolderName(datum.Folder), func(t *testing.T) {
			t.Helper()
			tests.Run(t, r, datum)
		})
	}
}

func (r *runner) CodeLens(t *testing.T, uri span.URI, want []protocol.CodeLens) {
	if source.DetectLanguage("", uri.Filename()) != source.Mod {
		return
	}
	v, err := r.server.session.ViewOf(uri)
	if err != nil {
		t.Fatal(err)
	}
	got, err := mod.CodeLens(r.ctx, v.Snapshot(), uri)
	if err != nil {
		t.Fatal(err)
	}
	if diff := tests.DiffCodeLens(uri, want, got); diff != "" {
		t.Error(diff)
	}
}

func (r *runner) Diagnostics(t *testing.T, uri span.URI, want []*source.Diagnostic) {
	// Get the diagnostics for this view if we have not done it before.
	if r.diagnostics == nil {
		r.diagnostics = make(map[span.URI][]*source.Diagnostic)
		v := r.server.session.View(r.data.Config.Dir)
		// Always run diagnostics with analysis.
		reports, _ := r.server.diagnose(r.ctx, v.Snapshot(), true)
		for key, diags := range reports {
			r.diagnostics[key.id.URI] = diags
		}
	}
	got := r.diagnostics[uri]
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
			URI: protocol.URIFromSpanURI(uri),
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
			URI: protocol.URIFromSpanURI(uri),
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
			URI: protocol.URIFromSpanURI(uri),
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
			URI: protocol.URIFromSpanURI(uri),
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
		d := myers.ComputeEdits(uri, want, got)
		t.Errorf("import failed for %s: %s", filename, diff.ToUnified("want", "got", want, d))
	}
}

func (r *runner) SuggestedFix(t *testing.T, spn span.Span, actionKinds []string) {
	uri := spn.URI()
	view, err := r.server.session.ViewOf(uri)
	if err != nil {
		t.Fatal(err)
	}
	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	rng, err := m.Range(spn)
	if err != nil {
		t.Fatal(err)
	}
	// Get the diagnostics for this view if we have not done it before.
	if r.diagnostics == nil {
		r.diagnostics = make(map[span.URI][]*source.Diagnostic)
		// Always run diagnostics with analysis.
		reports, _ := r.server.diagnose(r.ctx, view.Snapshot(), true)
		for key, diags := range reports {
			r.diagnostics[key.id.URI] = diags
		}
	}
	var diag *source.Diagnostic
	for _, d := range r.diagnostics[uri] {
		// Compare the start positions rather than the entire range because
		// some diagnostics have a range with the same start and end position (8:1-8:1).
		// The current marker functionality prevents us from having a range of 0 length.
		if protocol.ComparePosition(d.Range.Start, rng.Start) == 0 {
			diag = d
			break
		}
	}
	if diag == nil {
		t.Fatalf("could not get any suggested fixes for %v", spn)
	}
	codeActionKinds := []protocol.CodeActionKind{}
	for _, k := range actionKinds {
		codeActionKinds = append(codeActionKinds, protocol.CodeActionKind(k))
	}
	actions, err := r.server.CodeAction(r.ctx, &protocol.CodeActionParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
		Context: protocol.CodeActionContext{
			Only:        codeActionKinds,
			Diagnostics: toProtocolDiagnostics([]*source.Diagnostic{diag}),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// TODO: This test should probably be able to handle multiple code actions.
	if len(actions) == 0 {
		t.Fatal("no code actions returned")
	}
	res, err := applyWorkspaceEdits(r, actions[0].Edit)
	if err != nil {
		t.Fatal(err)
	}
	for u, got := range res {
		fixed := string(r.data.Golden("suggestedfix_"+tests.SpanName(spn), u.Filename(), func() ([]byte, error) {
			return []byte(got), nil
		}))
		if fixed != got {
			t.Errorf("suggested fixes failed for %s, expected:\n%#v\ngot:\n%#v", u.Filename(), fixed, got)
		}
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
			t.Errorf("%s:\n%s", d.Src, tests.Diff(expectHover, hover.Contents.Value))
		}
	}
	if !d.OnlyHover {
		didSomething = true
		locURI := locs[0].URI.SpanURI()
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
		locURI := locs[i].URI.SpanURI()
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
			URI: protocol.URIFromSpanURI(uri),
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
		uri := span.URIFromURI(orderedURIs[i])
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
	if got == nil {
		if want.Text != "" { // expected an ident.
			t.Errorf("prepare rename failed for %v: got nil", src)
		}
		return
	}
	if got.Start == got.End {
		// Special case for 0-length ranges. Marks can't specify a 0-length range,
		// so just compare the start.
		if got.Start != want.Range.Start {
			t.Errorf("prepare rename failed: incorrect point, got %v want %v", got.Start, want.Range.Start)
		}
	} else {
		if protocol.CompareRange(*got, want.Range) != 0 {
			t.Errorf("prepare rename failed: incorrect range got %v want %v", *got, want.Range)
		}
	}
}

func applyWorkspaceEdits(r *runner, wedit protocol.WorkspaceEdit) (map[span.URI]string, error) {
	res := map[span.URI]string{}
	for _, docEdits := range wedit.DocumentChanges {
		uri := docEdits.TextDocument.URI.SpanURI()
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
			URI: protocol.URIFromSpanURI(uri),
		},
	}
	got, err := r.server.DocumentSymbol(r.ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(expectedSymbols) {
		t.Errorf("want %d top-level symbols in %v, got %d", len(expectedSymbols), uri, len(got))
		return
	}
	symbols := make([]protocol.DocumentSymbol, len(got))
	for i, s := range got {
		s, ok := s.(protocol.DocumentSymbol)
		if !ok {
			t.Fatalf("%v: wanted []DocumentSymbols but got %v", uri, got)
		}
		symbols[i] = s
	}
	if diff := tests.DiffSymbols(t, uri, expectedSymbols, symbols); diff != "" {
		t.Error(diff)
	}
}

func (r *runner) WorkspaceSymbols(t *testing.T, query string, expectedSymbols []protocol.SymbolInformation, dirs map[string]struct{}) {
	r.callWorkspaceSymbols(t, query, source.SymbolCaseInsensitive, dirs, expectedSymbols)
}

func (r *runner) FuzzyWorkspaceSymbols(t *testing.T, query string, expectedSymbols []protocol.SymbolInformation, dirs map[string]struct{}) {
	r.callWorkspaceSymbols(t, query, source.SymbolFuzzy, dirs, expectedSymbols)
}

func (r *runner) CaseSensitiveWorkspaceSymbols(t *testing.T, query string, expectedSymbols []protocol.SymbolInformation, dirs map[string]struct{}) {
	r.callWorkspaceSymbols(t, query, source.SymbolCaseSensitive, dirs, expectedSymbols)
}

func (r *runner) callWorkspaceSymbols(t *testing.T, query string, matcher source.SymbolMatcher, dirs map[string]struct{}, expectedSymbols []protocol.SymbolInformation) {
	t.Helper()

	original := r.server.session.Options()
	modified := original
	modified.SymbolMatcher = matcher
	r.server.session.SetOptions(modified)
	defer r.server.session.SetOptions(original)

	params := &protocol.WorkspaceSymbolParams{
		Query: query,
	}
	got, err := r.server.Symbol(r.ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	got = tests.FilterWorkspaceSymbols(got, dirs)
	if len(got) != len(expectedSymbols) {
		t.Errorf("want %d symbols, got %d", len(expectedSymbols), len(got))
		return
	}
	if diff := tests.DiffWorkspaceSymbols(expectedSymbols, got); diff != "" {
		t.Error(diff)
	}
}

func (r *runner) SignatureHelp(t *testing.T, spn span.Span, want *protocol.SignatureHelp) {
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
			URI: protocol.URIFromSpanURI(spn.URI()),
		},
		Position: loc.Range.Start,
	}
	params := &protocol.SignatureHelpParams{
		TextDocumentPositionParams: tdpp,
	}
	got, err := r.server.SignatureHelp(r.ctx, params)
	if err != nil {
		// Only fail if we got an error we did not expect.
		if want != nil {
			t.Fatal(err)
		}
		return
	}
	if want == nil {
		if got != nil {
			t.Errorf("expected no signature, got %v", got)
		}
		return
	}
	if got == nil {
		t.Fatalf("expected %v, got nil", want)
	}
	if diff := tests.DiffSignatures(spn, want, got); diff != "" {
		t.Error(diff)
	}
}

func (r *runner) Link(t *testing.T, uri span.URI, wantLinks []tests.Link) {
	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	got, err := r.server.DocumentLink(r.ctx, &protocol.DocumentLinkParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
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
		uri := span.URIFromPath(fname)
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
