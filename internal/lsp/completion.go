// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"sort"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
)

func (s *Server) completion(ctx context.Context, params *protocol.CompletionParams) (*protocol.CompletionList, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	options := view.Options()
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	options.Completion.FullDocumentation = options.HoverKind == source.FullDocumentation
	candidates, surrounding, err := source.Completion(ctx, view, f, params.Position, options.Completion)
	if err != nil {
		log.Print(ctx, "no completions found", tag.Of("At", params.Position), tag.Of("Failure", err))
	}
	if candidates == nil {
		return &protocol.CompletionList{
			Items: []protocol.CompletionItem{},
		}, nil
	}
	// We might need to adjust the position to account for the prefix.
	rng, err := surrounding.Range()
	if err != nil {
		return nil, err
	}
	// Sort the candidates by score, since that is not supported by LSP yet.
	sort.SliceStable(candidates, func(i, j int) bool {
		return candidates[i].Score > candidates[j].Score
	})

	// When using deep completions/fuzzy matching, report results as incomplete so
	// client fetches updated completions after every key stroke.
	incompleteResults := options.Completion.Deep || options.Completion.FuzzyMatching

	items := toProtocolCompletionItems(candidates, rng, options)

	if incompleteResults {
		prefix := surrounding.Prefix()
		for i := range items {
			// We send the prefix as the filterText to trick VSCode into not
			// reordering our candidates. All the candidates will appear to
			// be a perfect match, so VSCode's fuzzy matching/ranking just
			// maintains the natural "sortText" ordering. We can only do
			// this in tandem with "incompleteResults" since otherwise
			// client side filtering is important.
			items[i].FilterText = prefix
		}
	}

	return &protocol.CompletionList{
		IsIncomplete: incompleteResults,
		Items:        items,
	}, nil
}

func toProtocolCompletionItems(candidates []source.CompletionItem, rng protocol.Range, options source.Options) []protocol.CompletionItem {
	var (
		items                  = make([]protocol.CompletionItem, 0, len(candidates))
		numDeepCompletionsSeen int
	)
	for i, candidate := range candidates {
		// Limit the number of deep completions to not overwhelm the user in cases
		// with dozens of deep completion matches.
		if candidate.Depth > 0 {
			if !options.Completion.Deep {
				continue
			}
			if numDeepCompletionsSeen >= source.MaxDeepCompletions {
				continue
			}
			numDeepCompletionsSeen++
		}
		insertText := candidate.InsertText
		if options.InsertTextFormat == protocol.SnippetTextFormat {
			insertText = candidate.Snippet()
		}

		// This can happen if the client has snippets disabled but the
		// candidate only supports snippet insertion.
		if insertText == "" {
			continue
		}

		item := protocol.CompletionItem{
			Label:  candidate.Label,
			Detail: candidate.Detail,
			Kind:   candidate.Kind,
			TextEdit: &protocol.TextEdit{
				NewText: insertText,
				Range:   rng,
			},
			InsertTextFormat:    options.InsertTextFormat,
			AdditionalTextEdits: candidate.AdditionalTextEdits,
			// This is a hack so that the client sorts completion results in the order
			// according to their score. This can be removed upon the resolution of
			// https://github.com/Microsoft/language-server-protocol/issues/348.
			SortText:      fmt.Sprintf("%05d", i),
			FilterText:    candidate.InsertText,
			Preselect:     i == 0,
			Documentation: candidate.Documentation,
		}
		// Trigger signature help for any function or method completion.
		// This is helpful even if a function does not have parameters,
		// since we show return types as well.
		switch item.Kind {
		case protocol.FunctionCompletion, protocol.MethodCompletion:
			item.Command = &protocol.Command{
				Command: "editor.action.triggerParameterHints",
			}
		}
		items = append(items, item)
	}
	return items
}
