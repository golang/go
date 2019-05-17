// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (s *Server) completion(ctx context.Context, params *protocol.CompletionParams) (*protocol.CompletionList, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	f, m, err := getGoFile(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(params.Position)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	items, surrounding, err := source.Completion(ctx, f, rng.Start)
	if err != nil {
		s.session.Logger().Infof(ctx, "no completions found for %s:%v:%v: %v", uri, int(params.Position.Line), int(params.Position.Character), err)
	}
	// We might need to adjust the position to account for the prefix.
	insertionRng := protocol.Range{
		Start: params.Position,
		End:   params.Position,
	}
	var prefix string
	if surrounding != nil {
		prefix = surrounding.Prefix()
		spn, err := surrounding.Range.Span()
		if err != nil {
			s.session.Logger().Infof(ctx, "failed to get span for surrounding position: %s:%v:%v: %v", uri, int(params.Position.Line), int(params.Position.Character), err)
		} else {
			rng, err := m.Range(spn)
			if err != nil {
				s.session.Logger().Infof(ctx, "failed to convert surrounding position: %s:%v:%v: %v", uri, int(params.Position.Line), int(params.Position.Character), err)
			} else {
				insertionRng = rng
			}
		}
	}
	return &protocol.CompletionList{
		IsIncomplete: false,
		Items:        toProtocolCompletionItems(items, prefix, insertionRng, s.insertTextFormat, s.usePlaceholders),
	}, nil
}

func toProtocolCompletionItems(candidates []source.CompletionItem, prefix string, rng protocol.Range, insertTextFormat protocol.InsertTextFormat, usePlaceholders bool) []protocol.CompletionItem {
	sort.SliceStable(candidates, func(i, j int) bool {
		return candidates[i].Score > candidates[j].Score
	})
	items := make([]protocol.CompletionItem, 0, len(candidates))
	for i, candidate := range candidates {
		// Match against the label.
		if !strings.HasPrefix(candidate.Label, prefix) {
			continue
		}
		insertText := candidate.InsertText
		if insertTextFormat == protocol.SnippetTextFormat {
			insertText = candidate.Snippet(usePlaceholders)
		}
		item := protocol.CompletionItem{
			Label:  candidate.Label,
			Detail: candidate.Detail,
			Kind:   toProtocolCompletionItemKind(candidate.Kind),
			TextEdit: &protocol.TextEdit{
				NewText: insertText,
				Range:   rng,
			},
			InsertTextFormat: insertTextFormat,
			// This is a hack so that the client sorts completion results in the order
			// according to their score. This can be removed upon the resolution of
			// https://github.com/Microsoft/language-server-protocol/issues/348.
			SortText:   fmt.Sprintf("%05d", i),
			FilterText: candidate.InsertText,
			Preselect:  i == 0,
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

func toProtocolCompletionItemKind(kind source.CompletionItemKind) protocol.CompletionItemKind {
	switch kind {
	case source.InterfaceCompletionItem:
		return protocol.InterfaceCompletion
	case source.StructCompletionItem:
		return protocol.StructCompletion
	case source.TypeCompletionItem:
		return protocol.TypeParameterCompletion // ??
	case source.ConstantCompletionItem:
		return protocol.ConstantCompletion
	case source.FieldCompletionItem:
		return protocol.FieldCompletion
	case source.ParameterCompletionItem, source.VariableCompletionItem:
		return protocol.VariableCompletion
	case source.FunctionCompletionItem:
		return protocol.FunctionCompletion
	case source.MethodCompletionItem:
		return protocol.MethodCompletion
	case source.PackageCompletionItem:
		return protocol.ModuleCompletion // ??
	default:
		return protocol.TextCompletion
	}
}
