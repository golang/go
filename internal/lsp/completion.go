// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"strings"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func (s *Server) completion(ctx context.Context, params *protocol.CompletionParams) (*protocol.CompletionList, error) {
	snapshot, fh, ok, err := s.beginFileRequest(params.TextDocument.URI, source.UnknownKind)
	if !ok {
		return nil, err
	}
	var candidates []source.CompletionItem
	var surrounding *source.Selection
	switch fh.Identity().Kind {
	case source.Go:
		candidates, surrounding, err = source.Completion(ctx, snapshot, fh, params.Position)
	case source.Mod:
		candidates, surrounding = nil, nil
	}
	if err != nil {
		event.Error(ctx, "no completions found", err, tag.Position.Of(params.Position))
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

	// When using deep completions/fuzzy matching, report results as incomplete so
	// client fetches updated completions after every key stroke.
	options := snapshot.View().Options()
	incompleteResults := options.DeepCompletion || options.Matcher == source.Fuzzy

	items := toProtocolCompletionItems(candidates, rng, options)

	if incompleteResults {
		for i := 1; i < len(items); i++ {
			// Give all the candidates the same filterText to trick VSCode
			// into not reordering our candidates. All the candidates will
			// appear to be equally good matches, so VSCode's fuzzy
			// matching/ranking just maintains the natural "sortText"
			// ordering. We can only do this in tandem with
			// "incompleteResults" since otherwise client side filtering is
			// important.
			items[i].FilterText = items[0].FilterText
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
			if !options.DeepCompletion {
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
			SortText: fmt.Sprintf("%05d", i),

			// Trim operators (VSCode doesn't like weird characters in
			// filterText).
			FilterText: strings.TrimLeft(candidate.InsertText, "&*"),

			Preselect:     i == 0,
			Documentation: candidate.Documentation,
		}
		items = append(items, item)
	}
	return items
}
