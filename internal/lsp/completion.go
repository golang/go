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
	"golang.org/x/tools/internal/span"
)

func (s *Server) completion(ctx context.Context, params *protocol.CompletionParams) (*protocol.CompletionList, error) {
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	defer release()
	if !ok {
		return nil, err
	}
	var candidates []source.CompletionItem
	var surrounding *source.Selection
	switch fh.Kind() {
	case source.Go:
		candidates, surrounding, err = source.Completion(ctx, snapshot, fh, params.Position, params.Context.TriggerCharacter)
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

	// internal/span treats end of file as the beginning of the next line, even
	// when it's not newline-terminated. We correct for that behaviour here if
	// end of file is not newline-terminated. See golang/go#41029.
	src, err := fh.Read()
	if err != nil {
		return nil, err
	}

	// Get the actual number of lines in source.
	numLines := len(strings.Split(string(src), "\n"))

	tok := snapshot.FileSet().File(surrounding.Start())
	endOfFile := tok.Pos(tok.Size())

	// For newline-terminated files, the line count reported by go/token should
	// be lower than the actual number of lines we see when splitting by \n. If
	// they're the same, the file isn't newline-terminated.
	if numLines == tok.LineCount() && tok.Size() != 0 {
		// Get span for character before end of file to bypass span's treatment of end
		// of file. We correct for this later.
		spn, err := span.NewRange(snapshot.FileSet(), endOfFile-1, endOfFile-1).Span()
		if err != nil {
			return nil, err
		}
		m := &protocol.ColumnMapper{
			URI:       fh.URI(),
			Converter: span.NewContentConverter(fh.URI().Filename(), []byte(src)),
			Content:   []byte(src),
		}
		eofRng, err := m.Range(spn)
		if err != nil {
			return nil, err
		}
		eofPosition := protocol.Position{
			Line: eofRng.Start.Line,
			// Correct for using endOfFile - 1 earlier.
			Character: eofRng.Start.Character + 1,
		}
		if surrounding.Start() == endOfFile {
			rng.Start = eofPosition
		}
		if surrounding.End() == endOfFile {
			rng.End = eofPosition
		}
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
