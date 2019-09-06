package source

import (
	"context"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/lsp/protocol"
)

func getCodeActions(ctx context.Context, view View, diag analysis.Diagnostic) ([]SuggestedFix, error) {
	var fixes []SuggestedFix
	for _, fix := range diag.SuggestedFixes {
		var edits []protocol.TextEdit
		for _, e := range fix.TextEdits {
			mrng, err := posToRange(ctx, view, e.Pos, e.End)
			if err != nil {
				return nil, err
			}
			rng, err := mrng.Range()
			if err != nil {
				return nil, err
			}
			edits = append(edits, protocol.TextEdit{
				Range:   rng,
				NewText: string(e.NewText),
			})
		}
		fixes = append(fixes, SuggestedFix{
			Title: fix.Message,
			Edits: edits,
		})
	}
	return fixes, nil
}
