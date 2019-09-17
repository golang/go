package source

import (
	"context"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

func getCodeActions(ctx context.Context, view View, pkg Package, diag analysis.Diagnostic) ([]SuggestedFix, error) {
	var fixes []SuggestedFix
	for _, fix := range diag.SuggestedFixes {
		var edits []protocol.TextEdit
		for _, e := range fix.TextEdits {
			posn := view.Session().Cache().FileSet().Position(e.Pos)
			ph, _, err := pkg.FindFile(ctx, span.FileURI(posn.Filename))
			if err != nil {
				return nil, err
			}
			_, m, _, err := ph.Cached(ctx)
			if err != nil {
				return nil, err
			}
			mrng, err := posToRange(ctx, view, m, e.Pos, e.End)
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
