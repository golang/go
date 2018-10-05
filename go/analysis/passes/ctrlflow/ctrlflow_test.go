package ctrlflow_test

import (
	"go/ast"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/ctrlflow"
)

func Test(t *testing.T) {
	testdata := analysistest.TestData()

	// load testdata/src/a/a.go
	results := analysistest.Run(t, testdata, ctrlflow.Analyzer, "a")

	// Perform a minimal smoke test on
	// the result (CFG) computed by ctrlflow.
	for _, result := range results {
		cfgs := result.Result.(*ctrlflow.CFGs)

		for _, decl := range result.Pass.Files[0].Decls {
			if decl, ok := decl.(*ast.FuncDecl); ok && decl.Body != nil {
				if cfgs.FuncDecl(decl) == nil {
					t.Errorf("%s: no CFG for func %s",
						result.Pass.Fset.Position(decl.Pos()), decl.Name.Name)
				}
			}
		}
	}
}
