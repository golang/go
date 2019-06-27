// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (s *Server) formatting(ctx context.Context, params *protocol.DocumentFormattingParams) ([]protocol.TextEdit, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	spn := span.New(uri, span.Point{}, span.Point{})
	f, m, rng, err := spanToRange(ctx, view, spn)
	if err != nil {
		return nil, err
	}
	edits, err := source.Format(ctx, f, rng)
	if err != nil {
		return nil, err
	}
	return ToProtocolEdits(m, edits)
}

func spanToRange(ctx context.Context, view source.View, s span.Span) (source.GoFile, *protocol.ColumnMapper, span.Range, error) {
	f, m, err := getGoFile(ctx, view, s.URI())
	if err != nil {
		return nil, nil, span.Range{}, err
	}
	rng, err := s.Range(m.Converter)
	if err != nil {
		return nil, nil, span.Range{}, err
	}
	if rng.Start == rng.End {
		// If we have a single point, assume we want the whole file.
		tok := f.GetToken(ctx)
		if tok == nil {
			return nil, nil, span.Range{}, fmt.Errorf("no file information for %s", f.URI())
		}
		rng.End = tok.Pos(tok.Size())
	}
	return f, m, rng, nil
}

func ToProtocolEdits(m *protocol.ColumnMapper, edits []source.TextEdit) ([]protocol.TextEdit, error) {
	if edits == nil {
		return nil, nil
	}
	result := make([]protocol.TextEdit, len(edits))
	for i, edit := range edits {
		rng, err := m.Range(edit.Span)
		if err != nil {
			return nil, err
		}
		result[i] = protocol.TextEdit{
			Range:   rng,
			NewText: edit.NewText,
		}
	}
	return result, nil
}

func FromProtocolEdits(m *protocol.ColumnMapper, edits []protocol.TextEdit) ([]source.TextEdit, error) {
	if edits == nil {
		return nil, nil
	}
	result := make([]source.TextEdit, len(edits))
	for i, edit := range edits {
		spn, err := m.RangeSpan(edit.Range)
		if err != nil {
			return nil, err
		}
		result[i] = source.TextEdit{
			Span:    spn,
			NewText: edit.NewText,
		}
	}
	return result, nil
}
