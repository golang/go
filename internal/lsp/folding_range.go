package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (s *Server) foldingRange(ctx context.Context, params *protocol.FoldingRangeParams) ([]protocol.FoldingRange, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return nil, err
	}
	snapshot := view.Snapshot()
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	ranges, err := source.FoldingRange(ctx, snapshot, f, view.Options().LineFoldingOnly)
	if err != nil {
		return nil, err
	}
	return toProtocolFoldingRanges(ranges)
}

func toProtocolFoldingRanges(ranges []*source.FoldingRangeInfo) ([]protocol.FoldingRange, error) {
	result := make([]protocol.FoldingRange, 0, len(ranges))
	for _, info := range ranges {
		rng, err := info.Range()
		if err != nil {
			return nil, err
		}
		result = append(result, protocol.FoldingRange{
			StartLine:      rng.Start.Line,
			StartCharacter: rng.Start.Character,
			EndLine:        rng.End.Line,
			EndCharacter:   rng.End.Character,
			Kind:           string(info.Kind),
		})
	}
	return result, nil
}
