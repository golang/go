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
	fh, err := snapshot.GetFile(uri)
	if err != nil {
		return nil, err
	}
	var ranges []*source.FoldingRangeInfo
	switch fh.Identity().Kind {
	case source.Go:
		ranges, err = source.FoldingRange(ctx, snapshot, fh, view.Options().LineFoldingOnly)
	case source.Mod:
		ranges = nil
	}

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
