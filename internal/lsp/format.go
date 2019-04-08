package lsp

import (
	"context"
	"fmt"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// formatRange formats a document with a given range.
func formatRange(ctx context.Context, v source.View, s span.Span) ([]protocol.TextEdit, error) {
	f, m, err := newColumnMap(ctx, v, s.URI())
	if err != nil {
		return nil, err
	}
	rng, err := s.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	if rng.Start == rng.End {
		// If we have a single point, assume we want the whole file.
		tok := f.GetToken(ctx)
		if tok == nil {
			return nil, fmt.Errorf("no file information for %s", f.URI())
		}
		rng.End = tok.Pos(tok.Size())
	}
	edits, err := source.Format(ctx, f, rng)
	if err != nil {
		return nil, err
	}
	return ToProtocolEdits(m, edits)
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

func newColumnMap(ctx context.Context, v source.View, uri span.URI) (source.File, *protocol.ColumnMapper, error) {
	f, err := v.GetFile(ctx, uri)
	if err != nil {
		return nil, nil, err
	}
	tok := f.GetToken(ctx)
	if tok == nil {
		return nil, nil, fmt.Errorf("no file information for %v", f.URI())
	}
	m := protocol.NewColumnMapper(f.URI(), f.GetFileSet(ctx), tok, f.GetContent(ctx))
	return f, m, nil
}
