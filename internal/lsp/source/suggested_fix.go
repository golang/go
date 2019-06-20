// +build !experimental

package source

import "go/token"
import "golang.org/x/tools/go/analysis"

func getCodeActions(fset *token.FileSet, diag analysis.Diagnostic) ([]SuggestedFixes, error) {
	return nil, nil
}
