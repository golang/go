// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source_test

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/fuzzy"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/testenv"
	errors "golang.org/x/xerrors"
)

func TestMain(m *testing.M) {
	testenv.ExitIfSmallMachine()
	os.Exit(m.Run())
}

func TestSource(t *testing.T) {
	packagestest.TestAll(t, testSource)
}

type runner struct {
	view source.View
	data *tests.Data
	ctx  context.Context
}

func testSource(t *testing.T, exporter packagestest.Exporter) {
	ctx := tests.Context(t)
	data := tests.Load(t, exporter, "../testdata")
	defer data.Exported.Cleanup()

	cache := cache.New(nil)
	session := cache.NewSession(ctx)
	options := tests.DefaultOptions()
	options.Env = data.Config.Env
	r := &runner{
		view: session.NewView(ctx, "source_test", span.FileURI(data.Config.Dir), options),
		data: data,
		ctx:  ctx,
	}
	for filename, content := range data.Config.Overlay {
		session.SetOverlay(span.FileURI(filename), source.DetectLanguage("", filename), content)
	}
	tests.Run(t, r, data)
}

func (r *runner) Diagnostics(t *testing.T, uri span.URI, want []source.Diagnostic) {
	f, err := r.view.GetFile(r.ctx, uri)
	if err != nil {
		t.Fatal(err)
	}
	results, _, err := source.Diagnostics(r.ctx, r.view, f, nil)
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

func (r *runner) Completion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	var want []protocol.CompletionItem
	for _, pos := range test.CompletionItems {
		want = append(want, tests.ToProtocolCompletionItem(*items[pos]))
	}
	prefix, list := r.callCompletion(t, src, source.CompletionOptions{
		Documentation: true,
		FuzzyMatching: true,
	})
	if !strings.Contains(string(src.URI()), "builtins") {
		list = tests.FilterBuiltins(list)
	}
	var got []protocol.CompletionItem
	for _, item := range list {
		if !strings.HasPrefix(strings.ToLower(item.Label), prefix) {
			continue
		}
		got = append(got, item)
	}
	if diff := tests.DiffCompletionItems(want, got); diff != "" {
		t.Errorf("%s: %s", src, diff)
	}
}

func (r *runner) CompletionSnippet(t *testing.T, src span.Span, expected tests.CompletionSnippet, placeholders bool, items tests.CompletionItems) {
	_, list := r.callCompletion(t, src, source.CompletionOptions{
		Placeholders: placeholders,
		Deep:         true,
	})
	got := tests.FindItem(list, *items[expected.CompletionItem])
	want := expected.PlainSnippet
	if placeholders {
		want = expected.PlaceholderSnippet
	}
	if diff := tests.DiffSnippets(want, got); diff != "" {
		t.Errorf("%s: %s", src, diff)
	}
}

func (r *runner) UnimportedCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	var want []protocol.CompletionItem
	for _, pos := range test.CompletionItems {
		want = append(want, tests.ToProtocolCompletionItem(*items[pos]))
	}
	_, got := r.callCompletion(t, src, source.CompletionOptions{
		Unimported: true,
	})
	if !strings.Contains(string(src.URI()), "builtins") {
		got = tests.FilterBuiltins(got)
	}
	if diff := tests.CheckCompletionOrder(want, got); diff != "" {
		t.Errorf("%s: %s", src, diff)
	}
}

func (r *runner) DeepCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	var want []protocol.CompletionItem
	for _, pos := range test.CompletionItems {
		want = append(want, tests.ToProtocolCompletionItem(*items[pos]))
	}
	prefix, list := r.callCompletion(t, src, source.CompletionOptions{
		Deep:          true,
		Documentation: true,
	})
	if !strings.Contains(string(src.URI()), "builtins") {
		list = tests.FilterBuiltins(list)
	}
	fuzzyMatcher := fuzzy.NewMatcher(prefix)
	var got []protocol.CompletionItem
	for _, item := range list {
		if fuzzyMatcher.Score(item.Label) <= 0 {
			continue
		}
		got = append(got, item)
	}
	if msg := tests.DiffCompletionItems(want, got); msg != "" {
		t.Errorf("%s: %s", src, msg)
	}
}

func (r *runner) FuzzyCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	var want []protocol.CompletionItem
	for _, pos := range test.CompletionItems {
		want = append(want, tests.ToProtocolCompletionItem(*items[pos]))
	}
	prefix, list := r.callCompletion(t, src, source.CompletionOptions{
		FuzzyMatching: true,
		Deep:          true,
	})
	if !strings.Contains(string(src.URI()), "builtins") {
		list = tests.FilterBuiltins(list)
	}
	var fuzzyMatcher *fuzzy.Matcher
	if prefix != "" {
		fuzzyMatcher = fuzzy.NewMatcher(prefix)
	}
	var got []protocol.CompletionItem
	for _, item := range list {
		if fuzzyMatcher != nil && fuzzyMatcher.Score(item.Label) <= 0 {
			continue
		}
		got = append(got, item)
	}
	if msg := tests.DiffCompletionItems(want, got); msg != "" {
		t.Errorf("%s: %s", src, msg)
	}
}

func (r *runner) CaseSensitiveCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	var want []protocol.CompletionItem
	for _, pos := range test.CompletionItems {
		want = append(want, tests.ToProtocolCompletionItem(*items[pos]))
	}
	_, list := r.callCompletion(t, src, source.CompletionOptions{
		CaseSensitive: true,
	})
	if !strings.Contains(string(src.URI()), "builtins") {
		list = tests.FilterBuiltins(list)
	}
	if diff := tests.DiffCompletionItems(want, list); diff != "" {
		t.Errorf("%s: %s", src, diff)
	}
}

func (r *runner) RankCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	var want []protocol.CompletionItem
	for _, pos := range test.CompletionItems {
		want = append(want, tests.ToProtocolCompletionItem(*items[pos]))
	}
	prefix, list := r.callCompletion(t, src, source.CompletionOptions{
		FuzzyMatching: true,
		Deep:          true,
	})
	fuzzyMatcher := fuzzy.NewMatcher(prefix)
	var got []protocol.CompletionItem
	for _, item := range list {
		if fuzzyMatcher.Score(item.Label) <= 0 {
			continue
		}
		got = append(got, item)
	}
	if msg := tests.CheckCompletionOrder(want, got); msg != "" {
		t.Errorf("%s: %s", src, msg)
	}
}

func (r *runner) callCompletion(t *testing.T, src span.Span, options source.CompletionOptions) (string, []protocol.CompletionItem) {
	f, err := r.view.GetFile(r.ctx, src.URI())
	if err != nil {
		t.Fatal(err)
	}
	list, surrounding, err := source.Completion(r.ctx, r.view, f, protocol.Position{
		Line:      float64(src.Start().Line() - 1),
		Character: float64(src.Start().Column() - 1),
	}, options)
	if err != nil && !errors.As(err, &source.ErrIsDefinition{}) {
		t.Fatalf("failed for %v: %v", src, err)
	}
	var prefix string
	if surrounding != nil {
		prefix = strings.ToLower(surrounding.Prefix())
	}
	// TODO(rstambler): In testing this out, I noticed that scores are equal,
	// even when they shouldn't be. This needs more investigation.
	sort.SliceStable(list, func(i, j int) bool {
		return list[i].Score > list[j].Score
	})
	var numDeepCompletionsSeen int
	var items []source.CompletionItem
	// Apply deep completion filtering.
	for _, item := range list {
		if item.Depth > 0 {
			if !options.Deep {
				continue
			}
			if numDeepCompletionsSeen >= source.MaxDeepCompletions {
				continue
			}
			numDeepCompletionsSeen++
		}
		items = append(items, item)
	}
	return prefix, tests.ToProtocolCompletionItems(items)
}

func (r *runner) FoldingRange(t *testing.T, spn span.Span) {
	uri := spn.URI()

	f, err := r.view.GetFile(r.ctx, uri)
	if err != nil {
		t.Fatalf("failed for %v: %v", spn, err)
	}
	fh := r.view.Snapshot().Handle(r.ctx, f)
	data, _, err := fh.Read(r.ctx)
	if err != nil {
		t.Error(err)
		return
	}

	// Test all folding ranges.
	ranges, err := source.FoldingRange(r.ctx, r.view, f, false)
	if err != nil {
		t.Error(err)
		return
	}
	r.foldingRanges(t, "foldingRange", uri, string(data), ranges)

	// Test folding ranges with lineFoldingOnly
	ranges, err = source.FoldingRange(r.ctx, r.view, f, true)
	if err != nil {
		t.Error(err)
		return
	}
	r.foldingRanges(t, "foldingRange-lineFolding", uri, string(data), ranges)
}

func (r *runner) foldingRanges(t *testing.T, prefix string, uri span.URI, data string, ranges []*source.FoldingRangeInfo) {
	t.Helper()
	// Fold all ranges.
	nonOverlapping := nonOverlappingRanges(t, ranges)
	for i, rngs := range nonOverlapping {
		got, err := foldRanges(string(data), rngs)
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
		var kindOnly []*source.FoldingRangeInfo
		for _, fRng := range ranges {
			if fRng.Kind == kind {
				kindOnly = append(kindOnly, fRng)
			}
		}

		nonOverlapping := nonOverlappingRanges(t, kindOnly)
		for i, rngs := range nonOverlapping {
			got, err := foldRanges(string(data), rngs)
			if err != nil {
				t.Error(err)
				continue
			}
			tag := fmt.Sprintf("%s-%s-%d", prefix, kind, i)
			want := string(r.data.Golden(tag, uri.Filename(), func() ([]byte, error) {
				return []byte(got), nil
			}))

			if want != got {
				t.Errorf("%s: failed for %s, expected:\n%v\ngot:\n%v", tag, uri.Filename(), want, got)
			}
		}

	}
}

func nonOverlappingRanges(t *testing.T, ranges []*source.FoldingRangeInfo) (res [][]*source.FoldingRangeInfo) {
	for _, fRng := range ranges {
		setNum := len(res)
		for i := 0; i < len(res); i++ {
			canInsert := true
			for _, rng := range res[i] {
				if conflict(t, rng, fRng) {
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
			res = append(res, []*source.FoldingRangeInfo{})
		}
		res[setNum] = append(res[setNum], fRng)
	}
	return res
}

func conflict(t *testing.T, a, b *source.FoldingRangeInfo) bool {
	arng, err := a.Range()
	if err != nil {
		t.Fatal(err)
	}
	brng, err := b.Range()
	if err != nil {
		t.Fatal(err)
	}
	// a start position is <= b start positions
	return protocol.ComparePosition(arng.Start, brng.Start) <= 0 && protocol.ComparePosition(arng.End, brng.Start) > 0
}

func foldRanges(contents string, ranges []*source.FoldingRangeInfo) (string, error) {
	foldedText := "<>"
	res := contents
	// Apply the folds from the end of the file forward
	// to preserve the offsets.
	for i := len(ranges) - 1; i >= 0; i-- {
		fRange := ranges[i]
		spn, err := fRange.Span()
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
	ctx := r.ctx
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
	edits, err := source.Format(ctx, r.view, f)
	if err != nil {
		if gofmted != "" {
			t.Error(err)
		}
		return
	}
	fh := r.view.Snapshot().Handle(r.ctx, f)
	data, _, err := fh.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	m, err := r.data.Mapper(f.URI())
	if err != nil {
		t.Fatal(err)
	}
	diffEdits, err := source.FromProtocolEdits(m, edits)
	if err != nil {
		t.Error(err)
	}
	got := diff.ApplyEdits(string(data), diffEdits)
	if gofmted != got {
		t.Errorf("format failed for %s, expected:\n%v\ngot:\n%v", filename, gofmted, got)
	}
}

func (r *runner) Import(t *testing.T, spn span.Span) {
	ctx := r.ctx
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
	fh := r.view.Snapshot().Handle(r.ctx, f)
	edits, err := source.Imports(ctx, r.view, f)
	if err != nil {
		if goimported != "" {
			t.Error(err)
		}
		return
	}
	data, _, err := fh.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	m, err := r.data.Mapper(fh.Identity().URI)
	if err != nil {
		t.Fatal(err)
	}
	diffEdits, err := source.FromProtocolEdits(m, edits)
	if err != nil {
		t.Error(err)
	}
	got := diff.ApplyEdits(string(data), diffEdits)
	if goimported != got {
		t.Errorf("import failed for %s, expected:\n%v\ngot:\n%v", filename, goimported, got)
	}
}

func (r *runner) SuggestedFix(t *testing.T, spn span.Span) {
}

func (r *runner) Definition(t *testing.T, spn span.Span, d tests.Definition) {
	ctx := r.ctx
	f, err := r.view.GetFile(ctx, d.Src.URI())
	if err != nil {
		t.Fatalf("failed for %v: %v", d.Src, err)
	}
	_, srcRng, err := spanToRange(r.data, d.Src)
	if err != nil {
		t.Fatal(err)
	}
	ident, err := source.Identifier(ctx, r.view, f, srcRng.Start)
	if err != nil {
		t.Fatalf("failed for %v: %v", d.Src, err)
	}
	h, err := ident.Hover(ctx)
	if err != nil {
		t.Fatalf("failed for %v: %v", d.Src, err)
	}
	var hover string
	if h.Synopsis != "" {
		hover += h.Synopsis + "\n"
	}
	hover += h.Signature
	rng, err := ident.Range()
	if err != nil {
		t.Fatal(err)
	}
	if d.IsType {
		rng, err = ident.Type.Range()
		if err != nil {
			t.Fatal(err)
		}
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
		if _, defRng, err := spanToRange(r.data, d.Def); err != nil {
			t.Fatal(err)
		} else if rng != defRng {
			t.Errorf("for %v got %v want %v", d.Src, rng, d.Def)
		}
	} else {
		t.Errorf("no tests ran for %s", d.Src.URI())
	}
}

func (r *runner) Highlight(t *testing.T, name string, locations []span.Span) {
	ctx := r.ctx
	src := locations[0]
	m, srcRng, err := spanToRange(r.data, src)
	if err != nil {
		t.Fatal(err)
	}
	highlights, err := source.Highlight(ctx, r.view, src.URI(), srcRng.Start)
	if err != nil {
		t.Errorf("highlight failed for %s: %v", src.URI(), err)
	}
	if len(highlights) != len(locations) {
		t.Errorf("got %d highlights for %s, expected %d", len(highlights), name, len(locations))
	}
	for i, got := range highlights {
		want, err := m.Range(locations[i])
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("want %v, got %v\n", want, got)
		}
	}
}

func (r *runner) References(t *testing.T, src span.Span, itemList []span.Span) {
	ctx := r.ctx
	f, err := r.view.GetFile(ctx, src.URI())
	if err != nil {
		t.Fatalf("failed for %v: %v", src, err)
	}
	_, srcRng, err := spanToRange(r.data, src)
	if err != nil {
		t.Fatal(err)
	}
	ident, err := source.Identifier(ctx, r.view, f, srcRng.Start)
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
		refSpan, err := refInfo.Span()
		if err != nil {
			t.Fatal(err)
		}
		got[refSpan] = true
	}

	if len(got) != len(want) {
		t.Errorf("references failed: different lengths got %v want %v", len(got), len(want))
	}

	for spn := range got {
		if !want[spn] {
			t.Errorf("references failed: incorrect references got %v want locations %v", got, want)
		}
	}
}

func (r *runner) Rename(t *testing.T, spn span.Span, newText string) {
	ctx := r.ctx
	tag := fmt.Sprintf("%s-rename", newText)

	f, err := r.view.GetFile(ctx, spn.URI())
	if err != nil {
		t.Fatalf("failed for %v: %v", spn, err)
	}
	_, srcRng, err := spanToRange(r.data, spn)
	if err != nil {
		t.Fatal(err)
	}
	ident, err := source.Identifier(r.ctx, r.view, f, srcRng.Start)
	if err != nil {
		t.Error(err)
		return
	}
	changes, err := ident.Rename(r.ctx, r.view, newText)
	if err != nil {
		renamed := string(r.data.Golden(tag, spn.URI().Filename(), func() ([]byte, error) {
			return []byte(err.Error()), nil
		}))
		if err.Error() != renamed {
			t.Errorf("rename failed for %s, expected:\n%v\ngot:\n%v\n", newText, renamed, err)
		}
		return
	}

	var res []string
	for editSpn, edits := range changes {
		f, err := r.view.GetFile(ctx, editSpn)
		if err != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}
		fh := r.view.Snapshot().Handle(r.ctx, f)
		data, _, err := fh.Read(ctx)
		if err != nil {
			t.Fatal(err)
		}
		m, err := r.data.Mapper(fh.Identity().URI)
		if err != nil {
			t.Fatal(err)
		}
		filename := filepath.Base(editSpn.Filename())
		diffEdits, err := source.FromProtocolEdits(m, edits)
		if err != nil {
			t.Fatal(err)
		}
		contents := applyEdits(string(data), diffEdits)
		if len(changes) > 1 {
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

	renamed := string(r.data.Golden(tag, spn.URI().Filename(), func() ([]byte, error) {
		return []byte(got), nil
	}))

	if renamed != got {
		t.Errorf("rename failed for %s, expected:\n%v\ngot:\n%v", newText, renamed, got)
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

func (r *runner) PrepareRename(t *testing.T, src span.Span, want *source.PrepareItem) {
	ctx := context.Background()
	f, err := r.view.GetFile(ctx, src.URI())
	if err != nil {
		t.Fatalf("failed for %v: %v", src, err)
	}
	_, srcRng, err := spanToRange(r.data, src)
	if err != nil {
		t.Fatal(err)
	}
	// Find the identifier at the position.
	item, err := source.PrepareRename(ctx, r.view, f, srcRng.Start)
	if err != nil {
		if want.Text != "" { // expected an ident.
			t.Errorf("prepare rename failed for %v: got error: %v", src, err)
		}
		return
	}
	if item == nil {
		if want.Text != "" {
			t.Errorf("prepare rename failed for %v: got nil", src)
		}
		return
	}
	if want.Text == "" && item != nil {
		t.Errorf("prepare rename failed for %v: expected nil, got %v", src, item)
		return
	}
	if protocol.CompareRange(want.Range, item.Range) != 0 {
		t.Errorf("prepare rename failed: incorrect range got %v want %v", item.Range, want.Range)
	}
}

func (r *runner) Symbol(t *testing.T, uri span.URI, expectedSymbols []protocol.DocumentSymbol) {
	ctx := r.ctx
	f, err := r.view.GetFile(ctx, uri)
	if err != nil {
		t.Fatalf("failed for %v: %v", uri, err)
	}
	symbols, err := source.DocumentSymbols(ctx, r.view, f)
	if err != nil {
		t.Errorf("symbols failed for %s: %v", uri, err)
	}
	if len(symbols) != len(expectedSymbols) {
		t.Errorf("want %d top-level symbols in %v, got %d", len(expectedSymbols), uri, len(symbols))
		return
	}
	if diff := r.diffSymbols(t, uri, expectedSymbols, symbols); diff != "" {
		t.Error(diff)
	}
}

func (r *runner) diffSymbols(t *testing.T, uri span.URI, want, got []protocol.DocumentSymbol) string {
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
		if protocol.CompareRange(w.SelectionRange, g.SelectionRange) != 0 {
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
	ctx := r.ctx
	f, err := r.view.GetFile(ctx, spn.URI())
	if err != nil {
		t.Fatalf("failed for %v: %v", spn, err)
	}
	_, rng, err := spanToRange(r.data, spn)
	if err != nil {
		t.Fatal(err)
	}
	gotSignature, err := source.SignatureHelp(ctx, r.view, f, rng.Start)
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
		return
	}
	if diff := diffSignatures(spn, expectedSignature, gotSignature); diff != "" {
		t.Error(diff)
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

func (r *runner) Link(t *testing.T, uri span.URI, wantLinks []tests.Link) {
	// This is a pure LSP feature, no source level functionality to be tested.
}

func spanToRange(data *tests.Data, spn span.Span) (*protocol.ColumnMapper, protocol.Range, error) {
	m, err := data.Mapper(spn.URI())
	if err != nil {
		return nil, protocol.Range{}, err
	}
	srcRng, err := m.Range(spn)
	if err != nil {
		return nil, protocol.Range{}, err
	}
	return m, srcRng, nil
}
