package lsp

import (
	"strings"
	"testing"
	"time"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/span"
)

func (r *runner) Completion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	got := r.callCompletion(t, src, source.CompletionOptions{
		Deep:          false,
		FuzzyMatching: false,
		Documentation: true,
	})
	if !strings.Contains(string(src.URI()), "builtins") {
		got = tests.FilterBuiltins(got)
	}
	want := expected(t, test, items)
	if diff := tests.DiffCompletionItems(want, got); diff != "" {
		t.Errorf("%s: %s", src, diff)
	}
}

func (r *runner) CompletionSnippet(t *testing.T, src span.Span, expected tests.CompletionSnippet, placeholders bool, items tests.CompletionItems) {
	list := r.callCompletion(t, src, source.CompletionOptions{
		Placeholders:  placeholders,
		Deep:          true,
		Budget:        5 * time.Second,
		FuzzyMatching: true,
	})
	got := tests.FindItem(list, *items[expected.CompletionItem])
	want := expected.PlainSnippet
	if placeholders {
		want = expected.PlaceholderSnippet
	}
	if diff := tests.DiffSnippets(want, got); diff != "" {
		t.Errorf("%s: %v", src, diff)
	}
}

func (r *runner) UnimportedCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	got := r.callCompletion(t, src, source.CompletionOptions{
		Unimported: true,
	})
	if !strings.Contains(string(src.URI()), "builtins") {
		got = tests.FilterBuiltins(got)
	}
	want := expected(t, test, items)
	if diff := tests.DiffCompletionItems(want, got); diff != "" {
		t.Errorf("%s: %s", src, diff)
	}
}

func (r *runner) DeepCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	got := r.callCompletion(t, src, source.CompletionOptions{
		Deep:          true,
		Budget:        5 * time.Second,
		Documentation: true,
	})
	if !strings.Contains(string(src.URI()), "builtins") {
		got = tests.FilterBuiltins(got)
	}
	want := expected(t, test, items)
	if msg := tests.DiffCompletionItems(want, got); msg != "" {
		t.Errorf("%s: %s", src, msg)
	}
}

func (r *runner) FuzzyCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	got := r.callCompletion(t, src, source.CompletionOptions{
		FuzzyMatching: true,
		Deep:          true,
		Budget:        5 * time.Second,
	})
	if !strings.Contains(string(src.URI()), "builtins") {
		got = tests.FilterBuiltins(got)
	}
	want := expected(t, test, items)
	if msg := tests.DiffCompletionItems(want, got); msg != "" {
		t.Errorf("%s: %s", src, msg)
	}
}

func (r *runner) CaseSensitiveCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	got := r.callCompletion(t, src, source.CompletionOptions{
		CaseSensitive: true,
	})
	if !strings.Contains(string(src.URI()), "builtins") {
		got = tests.FilterBuiltins(got)
	}
	want := expected(t, test, items)
	if msg := tests.DiffCompletionItems(want, got); msg != "" {
		t.Errorf("%s: %s", src, msg)
	}
}

func (r *runner) RankCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	got := r.callCompletion(t, src, source.CompletionOptions{
		FuzzyMatching: true,
		Deep:          true,
		Budget:        5 * time.Second,
	})
	want := expected(t, test, items)
	if msg := tests.CheckCompletionOrder(want, got); msg != "" {
		t.Errorf("%s: %s", src, msg)
	}
}

func expected(t *testing.T, test tests.Completion, items tests.CompletionItems) []protocol.CompletionItem {
	t.Helper()

	var want []protocol.CompletionItem
	for _, pos := range test.CompletionItems {
		item := items[pos]
		want = append(want, tests.ToProtocolCompletionItem(*item))
	}
	return want
}
func (r *runner) callCompletion(t *testing.T, src span.Span, options source.CompletionOptions) []protocol.CompletionItem {
	t.Helper()

	view := r.server.session.ViewOf(src.URI())
	original := view.Options()
	modified := original
	modified.InsertTextFormat = protocol.SnippetTextFormat
	modified.Completion = options
	view.SetOptions(modified)
	defer view.SetOptions(original)

	list, err := r.server.Completion(r.ctx, &protocol.CompletionParams{
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
	return list.Items
}
