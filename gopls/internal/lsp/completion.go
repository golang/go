// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"strings"

	"golang.org/x/tools/gopls/internal/lsp/lsppos"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/source/completion"
	"golang.org/x/tools/gopls/internal/lsp/template"
	"golang.org/x/tools/gopls/internal/lsp/work"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
)

func (s *Server) completion(ctx context.Context, params *protocol.CompletionParams) (*protocol.CompletionList, error) {
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	defer release()
	if !ok {
		return nil, err
	}
	var candidates []completion.CompletionItem
	var surrounding *completion.Selection
	switch snapshot.View().FileKind(fh) {
	case source.Go:
		candidates, surrounding, err = completion.Completion(ctx, snapshot, fh, params.Position, params.Context)
	case source.Mod:
		candidates, surrounding = nil, nil
	case source.Work:
		cl, err := work.Completion(ctx, snapshot, fh, params.Position)
		if err != nil {
			break
		}
		return cl, nil
	case source.Tmpl:
		var cl *protocol.CompletionList
		cl, err = template.Completion(ctx, snapshot, fh, params.Position, params.Context)
		if err != nil {
			break // use common error handling, candidates==nil
		}
		return cl, nil
	}
	if err != nil {
		event.Error(ctx, "no completions found", err, tag.Position.Of(params.Position))
	}
	if candidates == nil {
		return &protocol.CompletionList{
			IsIncomplete: true,
			Items:        []protocol.CompletionItem{},
		}, nil
	}

	// Map positions to LSP positions using the original content, rather than
	// internal/span, as the latter treats end of file as the beginning of the
	// next line, even when it's not newline-terminated. See golang/go#41029 for
	// more details.
	src, err := fh.Read()
	if err != nil {
		return nil, err
	}
	srng := surrounding.Range()
	tf := snapshot.FileSet().File(srng.Start) // not same as srng.TokFile due to //line
	rng, err := lsppos.NewTokenMapper(src, tf).Range(srng.Start, srng.End)
	if err != nil {
		return nil, err
	}

	// When using deep completions/fuzzy matching, report results as incomplete so
	// client fetches updated completions after every key stroke.
	options := snapshot.View().Options()
	incompleteResults := options.DeepCompletion || options.Matcher == source.Fuzzy

	items := toProtocolCompletionItems(candidates, rng, options)

	return &protocol.CompletionList{
		IsIncomplete: incompleteResults,
		Items:        items,
	}, nil
}

func toProtocolCompletionItems(candidates []completion.CompletionItem, rng protocol.Range, options *source.Options) []protocol.CompletionItem {
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
			if numDeepCompletionsSeen >= completion.MaxDeepCompletions {
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
			Tags:          candidate.Tags,
			Deprecated:    candidate.Deprecated,
		}
		items = append(items, item)
	}
	return items
}
