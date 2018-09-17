// The findcall package is a trivial example and test of an analyzer of
// Go source code. It reports a finding for every call to a function or
// method of the name specified by its --name flag.
package findcall

import (
	"go/ast"

	"golang.org/x/tools/go/analysis"
)

var Analysis = &analysis.Analysis{
	Name:             "findcall",
	Doc:              "find calls to a particular function",
	Run:              findcall,
	RunDespiteErrors: true,
}

var name = "println" // --name flag

func init() {
	Analysis.Flags.StringVar(&name, "name", name, "name of the function to find")
}

func findcall(unit *analysis.Unit) error {
	for _, f := range unit.Syntax {
		ast.Inspect(f, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				var id *ast.Ident
				switch fun := call.Fun.(type) {
				case *ast.Ident:
					id = fun
				case *ast.SelectorExpr:
					id = fun.Sel
				}
				if id != nil && !unit.Info.Types[id].IsType() && id.Name == name {
					unit.Findingf(call.Lparen, "call of %s(...)", id.Name)
				}
			}
			return true
		})
	}

	return nil
}
