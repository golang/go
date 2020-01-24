package lsp

import (
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/span"
)

func (r *runner) Completion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	got := r.callCompletion(t, src, func(opts *source.Options) {
		opts.DeepCompletion = false
		opts.Matcher = source.CaseInsensitive
		opts.UnimportedCompletion = false
		opts.InsertTextFormat = protocol.PlainTextTextFormat
		// Only enable literal completions if in the completion literals tests.
		// TODO(rstambler): Separate out literal completion tests.
		if strings.Contains(string(src.URI()), "literal") {
			opts.InsertTextFormat = protocol.SnippetTextFormat
		}

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
	list := r.callCompletion(t, src, func(opts *source.Options) {
		opts.Placeholders = placeholders
		opts.DeepCompletion = true
		opts.Matcher = source.Fuzzy
		opts.UnimportedCompletion = false
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
	got := r.callCompletion(t, src, func(opts *source.Options) {})
	if !strings.Contains(string(src.URI()), "builtins") {
		got = tests.FilterBuiltins(got)
	}
	want := expected(t, test, items)
	if diff := tests.CheckCompletionOrder(want, got, false); diff != "" {
		t.Errorf("%s: %s", src, diff)
	}
}

func (r *runner) DeepCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	got := r.callCompletion(t, src, func(opts *source.Options) {
		opts.DeepCompletion = true
		opts.Matcher = source.CaseInsensitive
		opts.UnimportedCompletion = false
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
	got := r.callCompletion(t, src, func(opts *source.Options) {
		opts.DeepCompletion = true
		opts.Matcher = source.Fuzzy
		opts.UnimportedCompletion = false
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
	got := r.callCompletion(t, src, func(opts *source.Options) {
		opts.Matcher = source.CaseSensitive
		opts.UnimportedCompletion = false
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
	got := r.callCompletion(t, src, func(opts *source.Options) {
		opts.DeepCompletion = true
		opts.Matcher = source.Fuzzy
		opts.UnimportedCompletion = false
	})
	want := expected(t, test, items)
	if msg := tests.CheckCompletionOrder(want, got, true); msg != "" {
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

func (r *runner) callCompletion(t *testing.T, src span.Span, options func(*source.Options)) []protocol.CompletionItem {
	t.Helper()

	view, err := r.server.session.ViewOf(src.URI())
	if err != nil {
		t.Fatal(err)
	}
	original := view.Options()
	modified := original
	options(&modified)
	view, err = view.SetOptions(r.ctx, modified)
	if err != nil {
		t.Error(err)
		return nil
	}
	defer view.SetOptions(r.ctx, original)

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
