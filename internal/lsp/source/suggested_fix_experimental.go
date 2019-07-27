// +build experimental

package source

import (
	"go/token"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/span"
)

func getCodeActions(fset *token.FileSet, diag analysis.Diagnostic) ([]SuggestedFixes, error) {
	var cas []SuggestedFixes
	for _, fix := range diag.SuggestedFixes {
		var ca SuggestedFixes
		ca.Title = fix.Message
		for _, te := range fix.TextEdits {
			span, err := span.NewRange(fset, te.Pos, te.End).Span()
			if err != nil {
				return nil, err
			}
			ca.Edits = append(ca.Edits, TextEdit{span, string(te.NewText)})
		}
		cas = append(cas, ca)
	}
	return cas, nil
}
