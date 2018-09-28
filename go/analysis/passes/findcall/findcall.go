// The findcall package is a trivial example and test of an analyzer of
// Go source code. It reports a diagnostic for every call to a function or
// method of the name specified by its --name flag.
package findcall

import (
	"go/ast"

	"golang.org/x/tools/go/analysis"
)

var Analyzer = &analysis.Analyzer{
	Name: "findcall",
	Doc: `find calls to a particular function

The findcall analysis reports calls to functions or methods
of a particular name.`,
	Run:              findcall,
	RunDespiteErrors: true,
}

var name = "println" // -name flag

func init() {
	Analyzer.Flags.StringVar(&name, "name", name, "name of the function to find")
}

func findcall(pass *analysis.Pass) (interface{}, error) {
	for _, f := range pass.Files {
		ast.Inspect(f, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				var id *ast.Ident
				switch fun := call.Fun.(type) {
				case *ast.Ident:
					id = fun
				case *ast.SelectorExpr:
					id = fun.Sel
				}
				if id != nil && !pass.TypesInfo.Types[id].IsType() && id.Name == name {
					pass.Reportf(call.Lparen, "call of %s(...)", id.Name)
				}
			}
			return true
		})
	}

	return nil, nil
}
