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
	"time"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/fuzzy"
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

	cache := cache.New()
	session := cache.NewSession(ctx)
	options := session.Options()
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

func (r *runner) Diagnostics(t *testing.T, data tests.Diagnostics) {
	for uri, want := range data {
		f, err := r.view.GetFile(r.ctx, uri)
		if err != nil {
			t.Fatal(err)
		}
		results, _, err := source.Diagnostics(r.ctx, r.view, f.(source.GoFile), nil)
		if err != nil {
			t.Fatal(err)
		}
		got := results[uri]
		// A special case to test that there are no diagnostics for a file.
		if len(want) == 1 && want[0].Source == "no_diagnostics" {
			if len(got) != 0 {
				t.Errorf("expected no diagnostics for %s, got %v", uri, got)
			}
			continue
		}
		if diff := tests.DiffDiagnostics(uri, want, got); diff != "" {
			t.Error(diff)
		}
	}
}

func (r *runner) Completion(t *testing.T, data tests.Completions, snippets tests.CompletionSnippets, items tests.CompletionItems) {
	ctx := r.ctx
	for src, test := range data {
		var want []source.CompletionItem
		for _, pos := range test.CompletionItems {
			want = append(want, *items[pos])
		}
		f, err := r.view.GetFile(ctx, src.URI())
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}
		deepComplete := strings.Contains(string(src.URI()), "deepcomplete")
		fuzzyMatch := strings.Contains(string(src.URI()), "fuzzymatch")
		unimported := strings.Contains(string(src.URI()), "unimported")
		list, surrounding, err := source.Completion(ctx, r.view, f.(source.GoFile), protocol.Position{
			Line:      float64(src.Start().Line() - 1),
			Character: float64(src.Start().Column() - 1),
		}, source.CompletionOptions{
			Documentation: true,
			Deep:          deepComplete,
			FuzzyMatching: fuzzyMatch,
			Unimported:    unimported,
			// Crank this up so tests don't flake.
			Budget: 5 * time.Second,
		})
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}
		var (
			prefix       string
			fuzzyMatcher *fuzzy.Matcher
		)
		if surrounding != nil {
			prefix = strings.ToLower(surrounding.Prefix())
			if deepComplete && prefix != "" {
				fuzzyMatcher = fuzzy.NewMatcher(surrounding.Prefix(), fuzzy.Symbol)
			}
		}
		wantBuiltins := strings.Contains(string(src.URI()), "builtins")
		var got []source.CompletionItem
		for _, item := range list {
			if !wantBuiltins && isBuiltin(item) {
				continue
			}

			// If deep completion is enabled, we need to use the fuzzy matcher to match
			// the code's behavior.
			if deepComplete {
				if fuzzyMatcher != nil && fuzzyMatcher.Score(item.Label) < 0 {
					continue
				}
			} else {
				// We let the client do fuzzy matching, so we return all possible candidates.
				// To simplify testing, filter results with prefixes that don't match exactly.
				if !strings.HasPrefix(strings.ToLower(item.Label), prefix) {
					continue
				}
			}
			got = append(got, item)
		}
		switch test.Type {
		case tests.CompletionFull:
			if diff := diffCompletionItems(want, got); diff != "" {
				t.Errorf("%s: %s", src, diff)
			}
		case tests.CompletionPartial:
			if msg := checkCompletionOrder(want, got); msg != "" {
				t.Errorf("%s: %s", src, msg)
			}
		}
	}
	for _, usePlaceholders := range []bool{true, false} {
		for src, want := range snippets {
			f, err := r.view.GetFile(ctx, src.URI())
			if err != nil {
				t.Fatalf("failed for %v: %v", src, err)
			}

			list, _, err := source.Completion(ctx, r.view, f.(source.GoFile), protocol.Position{
				Line:      float64(src.Start().Line() - 1),
				Character: float64(src.Start().Column() - 1),
			}, source.CompletionOptions{
				Documentation: true,
				Deep:          strings.Contains(string(src.URI()), "deepcomplete"),
				FuzzyMatching: strings.Contains(string(src.URI()), "fuzzymatch"),
				Placeholders:  usePlaceholders,
				// Crank this up so tests don't flake.
				Budget: 5 * time.Second,
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
			expected := want.PlainSnippet
			if usePlaceholders {
				expected = want.PlaceholderSnippet
			}
			if expected == "" {
				if got != nil {
					t.Fatalf("%s:%d: expected no matching snippet", src.URI(), src.Start().Line())
				}
			} else {
				if got == nil {
					t.Fatalf("%s:%d: couldn't find completion matching %q", src.URI(), src.Start().Line(), wantItem.Label)
				}
				actual := got.Snippet()
				if expected != actual {
					t.Errorf("%s: expected placeholder snippet %q, got %q", src, expected, actual)
				}
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
func diffCompletionItems(want []source.CompletionItem, got []source.CompletionItem) string {
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
		if w.Documentation != "" && !strings.HasPrefix(w.Documentation, "@") {
			if w.Documentation != g.Documentation {
				return summarizeCompletionItems(i, want, got, "incorrect Documentation got %v want %v", g.Documentation, w.Documentation)
			}
		}
		if w.Kind != g.Kind {
			return summarizeCompletionItems(i, want, got, "incorrect Kind got %v want %v", g.Kind, w.Kind)
		}
	}
	return ""
}

func checkCompletionOrder(want []source.CompletionItem, got []source.CompletionItem) string {
	var (
		matchedIdxs []int
		lastGotIdx  int
		inOrder     = true
	)
	for _, w := range want {
		var found bool
		for i, g := range got {
			if w.Label == g.Label && w.Detail == g.Detail && w.Kind == g.Kind {
				matchedIdxs = append(matchedIdxs, i)
				found = true
				if i < lastGotIdx {
					inOrder = false
				}
				lastGotIdx = i
				break
			}
		}
		if !found {
			return summarizeCompletionItems(-1, []source.CompletionItem{w}, got, "didn't find expected completion")
		}
	}

	sort.Ints(matchedIdxs)
	matched := make([]source.CompletionItem, 0, len(matchedIdxs))
	for _, idx := range matchedIdxs {
		matched = append(matched, got[idx])
	}

	if !inOrder {
		return summarizeCompletionItems(-1, want, matched, "completions out of order")
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

func (r *runner) FoldingRange(t *testing.T, data tests.FoldingRanges) {
	for _, spn := range data {
		uri := spn.URI()

		f, err := r.view.GetFile(r.ctx, uri)
		if err != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}
		data, _, err := f.Handle(r.ctx).Read(r.ctx)
		if err != nil {
			t.Error(err)
			continue
		}

		// Test all folding ranges.
		ranges, err := source.FoldingRange(r.ctx, r.view, f.(source.GoFile), false)
		if err != nil {
			t.Error(err)
			continue
		}
		r.foldingRanges(t, "foldingRange", uri, string(data), ranges)

		// Test folding ranges with lineFoldingOnly
		ranges, err = source.FoldingRange(r.ctx, r.view, f.(source.GoFile), true)
		if err != nil {
			t.Error(err)
			continue
		}
		r.foldingRanges(t, "foldingRange-lineFolding", uri, string(data), ranges)

	}
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

func (r *runner) Format(t *testing.T, data tests.Formats) {
	ctx := r.ctx
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
		edits, err := source.Format(ctx, r.view, f)
		if err != nil {
			if gofmted != "" {
				t.Error(err)
			}
			continue
		}
		data, _, err := f.Handle(ctx).Read(ctx)
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
}

func (r *runner) Import(t *testing.T, data tests.Imports) {
	ctx := r.ctx
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
		fh := f.Handle(ctx)
		tok, err := r.view.Session().Cache().TokenHandle(fh).Token(ctx)
		if err != nil {
			t.Fatal(err)
		}
		rng, err := spn.Range(span.NewTokenConverter(r.data.Exported.ExpectFileSet, tok))
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
}

func (r *runner) SuggestedFix(t *testing.T, data tests.SuggestedFixes) {
}

func (r *runner) Definition(t *testing.T, data tests.Definitions) {
	ctx := r.ctx
	for _, d := range data {
		f, err := r.view.GetFile(ctx, d.Src.URI())
		if err != nil {
			t.Fatalf("failed for %v: %v", d.Src, err)
		}
		_, srcRng, err := spanToRange(r.data, d.Src)
		if err != nil {
			t.Fatal(err)
		}
		ident, err := source.Identifier(ctx, r.view, f.(source.GoFile), srcRng.Start)
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
}

func (r *runner) Highlight(t *testing.T, data tests.Highlights) {
	ctx := r.ctx
	for name, locations := range data {
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
}

func (r *runner) Reference(t *testing.T, data tests.References) {
	ctx := r.ctx
	for src, itemList := range data {
		f, err := r.view.GetFile(ctx, src.URI())
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}
		_, srcRng, err := spanToRange(r.data, src)
		if err != nil {
			t.Fatal(err)
		}
		ident, err := source.Identifier(ctx, r.view, f.(source.GoFile), srcRng.Start)
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

		for spn, _ := range got {
			if !want[spn] {
				t.Errorf("references failed: incorrect references got %v want locations %v", got, want)
			}
		}
	}
}

func (r *runner) Rename(t *testing.T, data tests.Renames) {
	ctx := r.ctx
	for spn, newText := range data {
		tag := fmt.Sprintf("%s-rename", newText)

		f, err := r.view.GetFile(ctx, spn.URI())
		if err != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}
		_, srcRng, err := spanToRange(r.data, spn)
		if err != nil {
			t.Fatal(err)
		}
		ident, err := source.Identifier(r.ctx, r.view, f.(source.GoFile), srcRng.Start)
		if err != nil {
			t.Error(err)
			continue
		}
		changes, err := ident.Rename(r.ctx, r.view, newText)
		if err != nil {
			renamed := string(r.data.Golden(tag, spn.URI().Filename(), func() ([]byte, error) {
				return []byte(err.Error()), nil
			}))
			if err.Error() != renamed {
				t.Errorf("rename failed for %s, expected:\n%v\ngot:\n%v\n", newText, renamed, err)
			}
			continue
		}

		var res []string
		for editSpn, edits := range changes {
			f, err := r.view.GetFile(ctx, editSpn)
			if err != nil {
				t.Fatalf("failed for %v: %v", spn, err)
			}
			fh := f.Handle(ctx)
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
			res = append(res, fmt.Sprintf("%s:\n%s", filename, contents))
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

func (r *runner) PrepareRename(t *testing.T, data tests.PrepareRenames) {
	ctx := context.Background()
	for src, want := range data {
		f, err := r.view.GetFile(ctx, src.URI())
		if err != nil {
			t.Fatalf("failed for %v: %v", src, err)
		}
		_, srcRng, err := spanToRange(r.data, src)
		if err != nil {
			t.Fatal(err)
		}
		// Find the identifier at the position.
		item, err := source.PrepareRename(ctx, r.view, f.(source.GoFile), srcRng.Start)
		if err != nil {
			if want.Text != "" { // expected an ident.
				t.Errorf("prepare rename failed for %v: got error: %v", src, err)
			}
			continue
		}
		if item == nil {
			if want.Text != "" {
				t.Errorf("prepare rename failed for %v: got nil", src)
			}
			continue
		}
		if want.Text == "" && item != nil {
			t.Errorf("prepare rename failed for %v: expected nil, got %v", src, item)
			continue
		}
		if protocol.CompareRange(want.Range, item.Range) != 0 {
			t.Errorf("prepare rename failed: incorrect range got %v want %v", item.Range, want.Range)
		}
	}
}

func (r *runner) Symbol(t *testing.T, data tests.Symbols) {
	ctx := r.ctx
	for uri, expectedSymbols := range data {
		f, err := r.view.GetFile(ctx, uri)
		if err != nil {
			t.Fatalf("failed for %v: %v", uri, err)
		}
		symbols, err := source.DocumentSymbols(ctx, r.view, f.(source.GoFile))
		if err != nil {
			t.Errorf("symbols failed for %s: %v", uri, err)
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

func (r *runner) SignatureHelp(t *testing.T, data tests.Signatures) {
	ctx := r.ctx
	for spn, expectedSignature := range data {
		f, err := r.view.GetFile(ctx, spn.URI())
		if err != nil {
			t.Fatalf("failed for %v: %v", spn, err)
		}
		_, rng, err := spanToRange(r.data, spn)
		if err != nil {
			t.Fatal(err)
		}
		gotSignature, err := source.SignatureHelp(ctx, r.view, f.(source.GoFile), rng.Start)
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
