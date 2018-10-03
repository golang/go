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
	pass, result := analysistest.Run(t, testdata, ctrlflow.Analyzer, "a")

	// Perform a minimal smoke test on
	// the result (CFG) computed by ctrlflow.
	if result != nil {
		cfgs := result.(*ctrlflow.CFGs)

		for _, decl := range pass.Files[0].Decls {
			if decl, ok := decl.(*ast.FuncDecl); ok && decl.Body != nil {
				if cfgs.FuncDecl(decl) == nil {
					t.Errorf("%s: no CFG for func %s",
						pass.Fset.Position(decl.Pos()), decl.Name.Name)
				}
			}
		}
	}
}
