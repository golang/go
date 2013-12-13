package imports

import (
	"lib" // @describe ref-pkg-import "lib"
)

// Tests that import another package.  (To make the tests run quickly,
// we avoid using imports in all the other tests.  Remember, each
// query causes parsing and typechecking of the whole program.)
//
// See go.tools/oracle/oracle_test.go for explanation.
// See imports.golden for expected query results.

var a int

func main() {
	const c = lib.Const // @describe ref-const "Const"
	lib.Func()          // @describe ref-func "Func"
	lib.Var++           // @describe ref-var "Var"
	var t lib.Type      // @describe ref-type "Type"
	p := t.Method(&a)   // @describe ref-method "Method"

	print(*p + 1) // @pointsto p "p "

	var _ lib.Type // @describe ref-pkg "lib"
}
