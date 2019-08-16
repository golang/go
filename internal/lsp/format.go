// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/diff"
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
	return source.ToProtocolEdits(m, edits)
}

func spanToRange(ctx context.Context, view source.View, spn span.Span) (source.GoFile, *protocol.ColumnMapper, span.Range, error) {
	f, err := getGoFile(ctx, view, spn.URI())
	if err != nil {
		return nil, nil, span.Range{}, err
	}
	m, err := getMapper(ctx, f)
	if err != nil {
		return nil, nil, span.Range{}, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, nil, span.Range{}, err
	}
	if rng.Start == rng.End {
		// If we have a single point, assume we want the whole file.
		tok, err := f.GetToken(ctx)
		if err != nil {
			return nil, nil, span.Range{}, err
		}
		rng.End = tok.Pos(tok.Size())
	}
	return f, m, rng, nil
}

func FromProtocolEdits(m *protocol.ColumnMapper, edits []protocol.TextEdit) ([]diff.TextEdit, error) {
	if edits == nil {
		return nil, nil
	}
	result := make([]diff.TextEdit, len(edits))
	for i, edit := range edits {
		spn, err := m.RangeSpan(edit.Range)
		if err != nil {
			return nil, err
		}
		result[i] = diff.TextEdit{
			Span:    spn,
			NewText: edit.NewText,
		}
	}
	return result, nil
}
