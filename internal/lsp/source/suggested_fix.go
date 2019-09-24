package source

import (
	"context"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

type SuggestedFix struct {
	Title string
	Edits map[span.URI][]protocol.TextEdit
}

func suggestedFixes(ctx context.Context, view View, pkg Package, diag analysis.Diagnostic) ([]SuggestedFix, error) {
	var fixes []SuggestedFix
	for _, fix := range diag.SuggestedFixes {
		edits := make(map[span.URI][]protocol.TextEdit)
		for _, e := range fix.TextEdits {
			posn := view.Session().Cache().FileSet().Position(e.Pos)
			uri := span.FileURI(posn.Filename)
			ph, _, err := pkg.FindFile(ctx, uri)
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
			edits[uri] = append(edits[uri], protocol.TextEdit{
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

// onlyDeletions returns true if all of the suggested fixes are deletions.
func onlyDeletions(fixes []SuggestedFix) bool {
	for _, fix := range fixes {
		for _, edits := range fix.Edits {
			for _, edit := range edits {
				if edit.NewText != "" {
					return false
				}
				if protocol.ComparePosition(edit.Range.Start, edit.Range.End) == 0 {
					return false
				}
			}
		}
	}
	return true
}
