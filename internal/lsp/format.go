package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// formatRange formats a document with a given range.
func formatRange(ctx context.Context, v source.View, s span.Span) ([]protocol.TextEdit, error) {
	f, m, err := newColumnMap(ctx, v, s.URI)
	if err != nil {
		return nil, err
	}
	rng := s.Range(m.Converter)
	if rng.Start == rng.End {
		// if we have a single point, then assume the rest of the file
		rng.End = f.GetToken(ctx).Pos(f.GetToken(ctx).Size())
	}
	edits, err := source.Format(ctx, f, rng)
	if err != nil {
		return nil, err
	}
	return toProtocolEdits(m, edits), nil
}

func toProtocolEdits(m *protocol.ColumnMapper, edits []source.TextEdit) []protocol.TextEdit {
	if edits == nil {
		return nil
	}
	result := make([]protocol.TextEdit, len(edits))
	for i, edit := range edits {
		result[i] = protocol.TextEdit{
			Range:   m.Range(edit.Span),
			NewText: edit.NewText,
		}
	}
	return result
}

func newColumnMap(ctx context.Context, v source.View, uri span.URI) (source.File, *protocol.ColumnMapper, error) {
	f, err := v.GetFile(ctx, uri)
	if err != nil {
		return nil, nil, err
	}
	m := protocol.NewColumnMapper(f.URI(), f.GetFileSet(ctx), f.GetToken(ctx), f.GetContent(ctx))
	return f, m, nil
}
