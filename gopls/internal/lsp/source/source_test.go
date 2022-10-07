// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/source/completion"
	"golang.org/x/tools/gopls/internal/lsp/tests"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/fuzzy"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	bug.PanicOnBugs = true
	testenv.ExitIfSmallMachine()
	os.Exit(m.Run())
}

func TestSource(t *testing.T) {
	tests.RunTests(t, "../testdata", true, testSource)
}

type runner struct {
	snapshot    source.Snapshot
	view        source.View
	data        *tests.Data
	ctx         context.Context
	normalizers []tests.Normalizer
}

func testSource(t *testing.T, datum *tests.Data) {
	ctx := tests.Context(t)

	cache := cache.New(nil, nil, nil)
	session := cache.NewSession(ctx)
	options := source.DefaultOptions().Clone()
	tests.DefaultOptions(options)
	options.SetEnvSlice(datum.Config.Env)
	view, _, release, err := session.NewView(ctx, "source_test", span.URIFromPath(datum.Config.Dir), options)
	if err != nil {
		t.Fatal(err)
	}
	release()
	defer view.Shutdown(ctx)

	// Enable type error analyses for tests.
	// TODO(golang/go#38212): Delete this once they are enabled by default.
	tests.EnableAllAnalyzers(view, options)
	view.SetOptions(ctx, options)

	var modifications []source.FileModification
	for filename, content := range datum.Config.Overlay {
		if filepath.Ext(filename) != ".go" {
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
	if err := session.ModifyFiles(ctx, modifications); err != nil {
		t.Fatal(err)
	}
	snapshot, release := view.Snapshot(ctx)
	defer release()
	r := &runner{
		view:        view,
		snapshot:    snapshot,
		data:        datum,
		ctx:         ctx,
		normalizers: tests.CollectNormalizers(datum.Exported),
	}
	tests.Run(t, r, datum)
}

func (r *runner) CallHierarchy(t *testing.T, spn span.Span, expectedCalls *tests.CallHierarchyResult) {
	mapper, err := r.data.Mapper(spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := mapper.Location(spn)
	if err != nil {
		t.Fatalf("failed for %v: %v", spn, err)
	}
	fh, err := r.snapshot.GetFile(r.ctx, spn.URI())
	if err != nil {
		t.Fatal(err)
	}

	items, err := source.PrepareCallHierarchy(r.ctx, r.snapshot, fh, loc.Range.Start)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) == 0 {
		t.Fatalf("expected call hierarchy item to be returned for identifier at %v\n", loc.Range)
	}

	callLocation := protocol.Location{
		URI:   items[0].URI,
		Range: items[0].Range,
	}
	if callLocation != loc {
		t.Fatalf("expected source.PrepareCallHierarchy to return identifier at %v but got %v\n", loc, callLocation)
	}

	incomingCalls, err := source.IncomingCalls(r.ctx, r.snapshot, fh, loc.Range.Start)
	if err != nil {
		t.Error(err)
	}
	var incomingCallItems []protocol.CallHierarchyItem
	for _, item := range incomingCalls {
		incomingCallItems = append(incomingCallItems, item.From)
	}
	msg := tests.DiffCallHierarchyItems(incomingCallItems, expectedCalls.IncomingCalls)
	if msg != "" {
		t.Error(fmt.Sprintf("incoming calls differ: %s", msg))
	}

	outgoingCalls, err := source.OutgoingCalls(r.ctx, r.snapshot, fh, loc.Range.Start)
	if err != nil {
		t.Error(err)
	}
	var outgoingCallItems []protocol.CallHierarchyItem
	for _, item := range outgoingCalls {
		outgoingCallItems = append(outgoingCallItems, item.To)
	}
	msg = tests.DiffCallHierarchyItems(outgoingCallItems, expectedCalls.OutgoingCalls)
	if msg != "" {
		t.Error(fmt.Sprintf("outgoing calls differ: %s", msg))
	}
}

func (r *runner) Diagnostics(t *testing.T, uri span.URI, want []*source.Diagnostic) {
	fileID, got, err := source.FileDiagnostics(r.ctx, r.snapshot, uri)
	if err != nil {
		t.Fatal(err)
	}
	tests.CompareDiagnostics(t, fileID.URI, want, got)
}

func (r *runner) Completion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	var want []protocol.CompletionItem
	for _, pos := range test.CompletionItems {
		want = append(want, tests.ToProtocolCompletionItem(*items[pos]))
	}
	_, got := r.callCompletion(t, src, func(opts *source.Options) {
		opts.Matcher = source.CaseInsensitive
		opts.DeepCompletion = false
		opts.CompleteUnimported = false
		opts.InsertTextFormat = protocol.SnippetTextFormat
		opts.LiteralCompletions = strings.Contains(string(src.URI()), "literal")
		opts.ExperimentalPostfixCompletions = strings.Contains(string(src.URI()), "postfix")
	})
	got = tests.FilterBuiltins(src, got)
	if diff := tests.DiffCompletionItems(want, got); diff != "" {
		t.Errorf("%s: %s", src, diff)
	}
}

func (r *runner) CompletionSnippet(t *testing.T, src span.Span, expected tests.CompletionSnippet, placeholders bool, items tests.CompletionItems) {
	_, list := r.callCompletion(t, src, func(opts *source.Options) {
		opts.UsePlaceholders = placeholders
		opts.DeepCompletion = true
		opts.CompleteUnimported = false
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
	_, got := r.callCompletion(t, src, func(opts *source.Options) {})
	got = tests.FilterBuiltins(src, got)
	if diff := tests.CheckCompletionOrder(want, got, false); diff != "" {
		t.Errorf("%s: %s", src, diff)
	}
}

func (r *runner) DeepCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	var want []protocol.CompletionItem
	for _, pos := range test.CompletionItems {
		want = append(want, tests.ToProtocolCompletionItem(*items[pos]))
	}
	prefix, list := r.callCompletion(t, src, func(opts *source.Options) {
		opts.DeepCompletion = true
		opts.Matcher = source.CaseInsensitive
		opts.CompleteUnimported = false
	})
	list = tests.FilterBuiltins(src, list)
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
	_, got := r.callCompletion(t, src, func(opts *source.Options) {
		opts.DeepCompletion = true
		opts.Matcher = source.Fuzzy
		opts.CompleteUnimported = false
	})
	got = tests.FilterBuiltins(src, got)
	if msg := tests.DiffCompletionItems(want, got); msg != "" {
		t.Errorf("%s: %s", src, msg)
	}
}

func (r *runner) CaseSensitiveCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	var want []protocol.CompletionItem
	for _, pos := range test.CompletionItems {
		want = append(want, tests.ToProtocolCompletionItem(*items[pos]))
	}
	_, list := r.callCompletion(t, src, func(opts *source.Options) {
		opts.Matcher = source.CaseSensitive
		opts.CompleteUnimported = false
	})
	list = tests.FilterBuiltins(src, list)
	if diff := tests.DiffCompletionItems(want, list); diff != "" {
		t.Errorf("%s: %s", src, diff)
	}
}

func (r *runner) RankCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	var want []protocol.CompletionItem
	for _, pos := range test.CompletionItems {
		want = append(want, tests.ToProtocolCompletionItem(*items[pos]))
	}
	_, got := r.callCompletion(t, src, func(opts *source.Options) {
		opts.DeepCompletion = true
		opts.Matcher = source.Fuzzy
		opts.ExperimentalPostfixCompletions = true
	})
	if msg := tests.CheckCompletionOrder(want, got, true); msg != "" {
		t.Errorf("%s: %s", src, msg)
	}
}

func (r *runner) callCompletion(t *testing.T, src span.Span, options func(*source.Options)) (string, []protocol.CompletionItem) {
	fh, err := r.snapshot.GetFile(r.ctx, src.URI())
	if err != nil {
		t.Fatal(err)
	}
	original := r.view.Options()
	modified := original.Clone()
	options(modified)
	newView, err := r.view.SetOptions(r.ctx, modified)
	if newView != r.view {
		t.Fatalf("options change unexpectedly created new view")
	}
	if err != nil {
		t.Fatal(err)
	}
	defer r.view.SetOptions(r.ctx, original)

	list, surrounding, err := completion.Completion(r.ctx, r.snapshot, fh, protocol.Position{
		Line:      uint32(src.Start().Line() - 1),
		Character: uint32(src.Start().Column() - 1),
	}, protocol.CompletionContext{})
	if err != nil && !errors.As(err, &completion.ErrIsDefinition{}) {
		t.Fatalf("failed for %v: %v", src, err)
	}
	var prefix string
	if surrounding != nil {
		prefix = strings.ToLower(surrounding.Prefix())
	}

	var numDeepCompletionsSeen int
	var items []completion.CompletionItem
	// Apply deep completion filtering.
	for _, item := range list {
		if item.Depth > 0 {
			if !modified.DeepCompletion {
				continue
			}
			if numDeepCompletionsSeen >= completion.MaxDeepCompletions {
				continue
			}
			numDeepCompletionsSeen++
		}
		items = append(items, item)
	}
	return prefix, tests.ToProtocolCompletionItems(items)
}

func (r *runner) FoldingRanges(t *testing.T, spn span.Span) {
	uri := spn.URI()

	fh, err := r.snapshot.GetFile(r.ctx, spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	data, err := fh.Read()
	if err != nil {
		t.Error(err)
		return
	}

	// Test all folding ranges.
	ranges, err := source.FoldingRange(r.ctx, r.snapshot, fh, false)
	if err != nil {
		t.Error(err)
		return
	}
	r.foldingRanges(t, "foldingRange", uri, string(data), ranges)

	// Test folding ranges with lineFoldingOnly
	ranges, err = source.FoldingRange(r.ctx, r.snapshot, fh, true)
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
		want := string(r.data.Golden(t, tag, uri.Filename(), func() ([]byte, error) {
			return []byte(got), nil
		}))

		if diff := compare.Text(want, got); diff != "" {
			t.Errorf("%s: foldingRanges failed for %s, diff:\n%v", tag, uri.Filename(), diff)
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
			want := string(r.data.Golden(t, tag, uri.Filename(), func() ([]byte, error) {
				return []byte(got), nil
			}))

			if diff := compare.Text(want, got); diff != "" {
				t.Errorf("%s: failed for %s, diff:\n%v", tag, uri.Filename(), diff)
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
	gofmted := string(r.data.Golden(t, "gofmt", spn.URI().Filename(), func() ([]byte, error) {
		cmd := exec.Command("gofmt", spn.URI().Filename())
		out, _ := cmd.Output() // ignore error, sometimes we have intentionally ungofmt-able files
		return out, nil
	}))
	fh, err := r.snapshot.GetFile(r.ctx, spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	edits, err := source.Format(r.ctx, r.snapshot, fh)
	if err != nil {
		if gofmted != "" {
			t.Error(err)
		}
		return
	}
	m, err := r.data.Mapper(spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	got, _, err := source.ApplyProtocolEdits(m, edits)
	if err != nil {
		t.Error(err)
	}
	if gofmted != got {
		t.Errorf("format failed for %s, expected:\n%v\ngot:\n%v", spn.URI().Filename(), gofmted, got)
	}
}

func (r *runner) SemanticTokens(t *testing.T, spn span.Span) {
	t.Skip("nothing to test in source")
}

func (r *runner) Import(t *testing.T, spn span.Span) {
	fh, err := r.snapshot.GetFile(r.ctx, spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	edits, _, err := source.AllImportsFixes(r.ctx, r.snapshot, fh)
	if err != nil {
		t.Error(err)
	}
	m, err := r.data.Mapper(fh.URI())
	if err != nil {
		t.Fatal(err)
	}
	got, _, err := source.ApplyProtocolEdits(m, edits)
	if err != nil {
		t.Error(err)
	}
	want := string(r.data.Golden(t, "goimports", spn.URI().Filename(), func() ([]byte, error) {
		return []byte(got), nil
	}))
	if d := compare.Text(got, want); d != "" {
		t.Errorf("import failed for %s:\n%s", spn.URI().Filename(), d)
	}
}

func (r *runner) Definition(t *testing.T, spn span.Span, d tests.Definition) {
	_, srcRng, err := spanToRange(r.data, d.Src)
	if err != nil {
		t.Fatal(err)
	}
	fh, err := r.snapshot.GetFile(r.ctx, spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	ident, err := source.Identifier(r.ctx, r.snapshot, fh, srcRng.Start)
	if err != nil {
		t.Fatalf("failed for %v: %v", d.Src, err)
	}
	h, err := source.HoverIdentifier(r.ctx, ident)
	if err != nil {
		t.Fatalf("failed for %v: %v", d.Src, err)
	}
	hover, err := source.FormatHover(h, r.view.Options())
	if err != nil {
		t.Fatal(err)
	}
	rng, err := ident.Declaration.MappedRange[0].Range()
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
	didSomething := false
	if hover != "" {
		didSomething = true
		tag := fmt.Sprintf("%s-hoverdef", d.Name)
		expectHover := string(r.data.Golden(t, tag, d.Src.URI().Filename(), func() ([]byte, error) {
			return []byte(hover), nil
		}))
		hover = tests.StripSubscripts(hover)
		expectHover = tests.StripSubscripts(expectHover)
		if hover != expectHover {
			tests.CheckSameMarkdown(t, hover, expectHover)

		}
	}
	if !d.OnlyHover {
		didSomething = true
		if _, defRng, err := spanToRange(r.data, d.Def); err != nil {
			t.Fatal(err)
		} else if rng != defRng {
			t.Errorf("for %v got %v want %v", d.Src, rng, defRng)
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
	fh, err := r.snapshot.GetFile(r.ctx, spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	locs, err := source.Implementation(r.ctx, r.snapshot, fh, loc.Range.Start)
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
	ctx := r.ctx
	m, srcRng, err := spanToRange(r.data, src)
	if err != nil {
		t.Fatal(err)
	}
	fh, err := r.snapshot.GetFile(r.ctx, src.URI())
	if err != nil {
		t.Fatal(err)
	}
	highlights, err := source.Highlight(ctx, r.snapshot, fh, srcRng.Start)
	if err != nil {
		t.Errorf("highlight failed for %s: %v", src.URI(), err)
	}
	if len(highlights) != len(locations) {
		t.Fatalf("got %d highlights for highlight at %v:%v:%v, expected %d", len(highlights), src.URI().Filename(), src.Start().Line(), src.Start().Column(), len(locations))
	}
	// Check to make sure highlights have a valid range.
	var results []span.Span
	for i := range highlights {
		h, err := m.RangeSpan(highlights[i])
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

func (r *runner) InlayHints(t *testing.T, src span.Span) {
	// TODO(golang/go#53315): add source test
}

func (r *runner) Hover(t *testing.T, src span.Span, text string) {
	ctx := r.ctx
	_, srcRng, err := spanToRange(r.data, src)
	if err != nil {
		t.Fatal(err)
	}
	fh, err := r.snapshot.GetFile(r.ctx, src.URI())
	if err != nil {
		t.Fatal(err)
	}
	hover, err := source.Hover(ctx, r.snapshot, fh, srcRng.Start)
	if err != nil {
		t.Errorf("hover failed for %s: %v", src.URI(), err)
	}
	if text == "" {
		if hover != nil {
			t.Errorf("want nil, got %v\n", hover)
		}
	} else {
		if hover == nil {
			t.Fatalf("want hover result to not be nil")
		}
		if got := hover.Contents.Value; got != text {
			t.Errorf("want %v, got %v\n", got, text)
		}
		if want, got := srcRng, hover.Range; want != got {
			t.Errorf("want range %v, got %v instead", want, got)
		}
	}
}

func (r *runner) References(t *testing.T, src span.Span, itemList []span.Span) {
	ctx := r.ctx
	_, srcRng, err := spanToRange(r.data, src)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := r.snapshot
	fh, err := snapshot.GetFile(r.ctx, src.URI())
	if err != nil {
		t.Fatal(err)
	}
	for _, includeDeclaration := range []bool{true, false} {
		t.Run(fmt.Sprintf("refs-declaration-%v", includeDeclaration), func(t *testing.T) {
			want := make(map[span.Span]bool)
			for i, pos := range itemList {
				// We don't want the first result if we aren't including the declaration.
				if i == 0 && !includeDeclaration {
					continue
				}
				want[pos] = true
			}
			refs, err := source.References(ctx, snapshot, fh, srcRng.Start, includeDeclaration)
			if err != nil {
				t.Fatalf("failed for %s: %v", src, err)
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
		})
	}
}

func (r *runner) Rename(t *testing.T, spn span.Span, newText string) {
	tag := fmt.Sprintf("%s-rename", newText)

	_, srcRng, err := spanToRange(r.data, spn)
	if err != nil {
		t.Fatal(err)
	}
	fh, err := r.snapshot.GetFile(r.ctx, spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	changes, _, err := source.Rename(r.ctx, r.snapshot, fh, srcRng.Start, newText)
	if err != nil {
		renamed := string(r.data.Golden(t, tag, spn.URI().Filename(), func() ([]byte, error) {
			return []byte(err.Error()), nil
		}))
		if err.Error() != renamed {
			t.Errorf("rename failed for %s, expected:\n%v\ngot:\n%v\n", newText, renamed, err)
		}
		return
	}

	var res []string
	for editURI, edits := range changes {
		fh, err := r.snapshot.GetFile(r.ctx, editURI)
		if err != nil {
			t.Fatal(err)
		}
		m, err := r.data.Mapper(fh.URI())
		if err != nil {
			t.Fatal(err)
		}
		contents, _, err := source.ApplyProtocolEdits(m, edits)
		if err != nil {
			t.Fatal(err)
		}
		if len(changes) > 1 {
			filename := filepath.Base(editURI.Filename())
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

	renamed := string(r.data.Golden(t, tag, spn.URI().Filename(), func() ([]byte, error) {
		return []byte(got), nil
	}))

	if renamed != got {
		t.Errorf("rename failed for %s, expected:\n%v\ngot:\n%v", newText, renamed, got)
	}
}

func (r *runner) PrepareRename(t *testing.T, src span.Span, want *source.PrepareItem) {
	// Removed in favor of just using the lsp_test implementation. See ../lsp_test.go
}

func (r *runner) Symbols(t *testing.T, uri span.URI, expectedSymbols []protocol.DocumentSymbol) {
	// Removed in favor of just using the lsp_test implementation. See ../lsp_test.go
}

func (r *runner) WorkspaceSymbols(t *testing.T, uri span.URI, query string, typ tests.WorkspaceSymbolsTestType) {
	// Removed in favor of just using the lsp_test implementation. See ../lsp_test.go
}

func (r *runner) SignatureHelp(t *testing.T, spn span.Span, want *protocol.SignatureHelp) {
	_, rng, err := spanToRange(r.data, spn)
	if err != nil {
		t.Fatal(err)
	}
	fh, err := r.snapshot.GetFile(r.ctx, spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	gotSignature, gotActiveParameter, err := source.SignatureHelp(r.ctx, r.snapshot, fh, rng.Start)
	if err != nil {
		// Only fail if we got an error we did not expect.
		if want != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}
		return
	}
	if gotSignature == nil {
		if want != nil {
			t.Fatalf("got nil signature, but expected %v", want)
		}
		return
	}
	got := &protocol.SignatureHelp{
		Signatures:      []protocol.SignatureInformation{*gotSignature},
		ActiveParameter: uint32(gotActiveParameter),
	}
	if diff := tests.DiffSignatures(spn, want, got); diff != "" {
		t.Error(diff)
	}
}

// These are pure LSP features, no source level functionality to be tested.
func (r *runner) Link(t *testing.T, uri span.URI, wantLinks []tests.Link)                          {}
func (r *runner) SuggestedFix(t *testing.T, spn span.Span, actions []tests.SuggestedFix, want int) {}
func (r *runner) FunctionExtraction(t *testing.T, start span.Span, end span.Span)                  {}
func (r *runner) MethodExtraction(t *testing.T, start span.Span, end span.Span)                    {}
func (r *runner) CodeLens(t *testing.T, uri span.URI, want []protocol.CodeLens)                    {}
func (r *runner) AddImport(t *testing.T, uri span.URI, expectedImport string)                      {}

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
